import os
from tqdm import tqdm
import flgo.algorithm.fedbase as fedbase
from abc import ABCMeta, abstractmethod
import copy
import torch
import flgo.utils.fmodule as fmodule

class AbstractProtocol(metaclass=ABCMeta):
    r"""
    Abstract Protocol
    """
    @abstractmethod
    def get_clients_for_iteration(self, *args, **kwarg):
        """Return clients that should perform iteration at the current moment"""
        pass

class BasicProtocol(AbstractProtocol, fedbase.BasicParty):
    def __init__(self, option):
        super().__init__()
        self.test_data = None
        self.val_data = None
        self.model = None
        self.clients = []
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_parallels = option['num_parallels']
        # server calculator
        self.device = self.gv.apply_for_device() if not option['server_with_cpu'] else torch.device('cpu')
        self.calculator = self.gv.TaskCalculator(self.device, optimizer_name=option['optimizer'])
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option
        self.id = -1

    def set_topology(self, adj):
        """
        Set the topology of clients that is defined by a N x N 2darray|list[list[int]]|Tensor. Two clients with adj[i,j]>0 is viewed reachable to each other

        Args:
            adj (ndarray|list|torch.Tensor): a NxN matrix
        """
        assert len(adj) == self.num_clients
        for ci in range(len(self.num_clients)):
            assert max(adj[ci])>0
            self.clients[ci].register_objects([self.clients[cj] for cj in range(len(self.num_clients)) if adj[ci][cj]>0 and cj!=ci], 'clients')
        self.topology = adj
        for c in self.clients: c.topology = adj

    def init_topology(self):
        """init topology for objects"""
        for ck,c in enumerate(self.clients):
            other_ids = [cid for cid in list(range(len(self.clients))) if ck!=cid]
            c_clients = [self.clients[cid] for cid in other_ids]
            c.register_objects(c_clients, 'clients')
            c.num_clients = len(c_clients)
        self.topology = 'full'

    def get_clients_for_iteration(self, *args, **kwarg):
        """Return clients that should update states at the current aggregation round

        Returns:
            clients (list): list of clients
        """
        return self.clients

    def _check_model(self):
        for c in self.clients:
            if c.model is None: c.model = copy.deepcopy(self.model)

    def run(self):
        for c in self.clients: c.init_state()
        self._check_model()
        self.gv.logger.time_start('Total Time Cost')
        if not self._load_checkpoint() and self.eval_interval>0:
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()
            self.gv.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()
                    self.gv.logger.time_end('Eval Time Cost')
                    if self.option.get('save_optimal', False): self.gv.logger.trace_optimal_state()
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
                if self.current_round % self.option.get('check_interval', 1) == 0: self._save_checkpoint()
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def test(self, model=None, flag:str='test'):
        r"""
        Evaluate the model on the test dataset owned by the server.

        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metrics (dict): the dict contains the evaluating results
        """
        if model is None: model = self.model
        dataset = getattr(self, flag+'_data') if hasattr(self, flag+'_data') else None
        if dataset is None:
            return {}
        else:
            if self.option['server_with_cpu']: model.to('cuda')
            if self.option['test_parallel'] and torch.cuda.device_count()>1:
                test_model = torch.nn.DataParallel(model.to('cuda'))
                self.calculator.device = torch.device('cuda')
            else:
                test_model = model
            res = self.calculator.test(test_model, dataset, batch_size=min(self.option['test_batch_size'], len(dataset)), num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])
            self.calculator.device = self.device
            model.to(self.device)
            return res

    def iterate(self):
        """The iteration over all decentralized clients. Return True or None if any clients' states were updated"""
        clients_for_iteration = self.get_clients_for_iteration()
        # compute next state for each client in Sc
        next_states = []
        for c in tqdm(clients_for_iteration, desc='Iterate Clients', leave=False): next_states.append(c.update_state())
        # flush the states of all the clients
        for c, cn in zip(clients_for_iteration, next_states): c.state = copy.deepcopy(cn)
        return

    @property
    def available_clients(self):
        """
        Return all the available clients at the current round.

        Returns:
            a list of indices of currently available clients
        """
        return [cid for cid in range(self.num_clients) if self.clients[cid].is_idle()]

    def register_clients(self, clients):
        """
        Regiser clients to self.clients, and update related attributes (e.g. self.num_clients)

        Args:
            clients (list): a list of objects
        """
        self.register_objects(clients, 'clients')
        self.num_clients = len(clients)
        for cid, c in enumerate(self.clients):
            c.client_id = cid
        for c in self.clients: c.register_server(self)
        self.clients_per_round = max(int(self.num_clients * self.proportion), 1)
        self.selected_clients = []
        self.dropped_clients = []

    def clear_clients(self):
        """
        Clear clients and debind each client and the server
        """
        clients = self.clear_objects('clients')
        self.num_clients = 0
        self.clients_per_round = 0
        for c in clients:
            c.clear_objects('server_list')
            if hasattr(c, 'server'): c.server = None
        return

    def reset_clients(self, clients: list):
        """
        Reset clients and update related settings (e.g. self, logger, and simulator)

        Args:
            clients (list): a list of objects
        """
        self.clear_clients()
        if self.gv.simulator is not None:
            self.gv.simulator.register_clients(clients)
            self.gv.simulator.initialize()
        self.register_clients(clients)
        objects = [self] + clients
        for ob in objects: ob.initialize()
        self.gv.logger.register_variable(coordinator=objects[0], participants=objects[1:], objects=objects)
        if self.gv.logger.scene == 'horizontal': self.gv.logger.register_variable(server=objects[0], clients=objects[1:])
        self.gv.logger.initialize()
        return

    def _save_checkpoint(self):
        checkpoint = self.option.get('save_checkpoint', '')
        if checkpoint != '':
            cpt = self.save_checkpoint()
            cpt_path = os.path.join(self.option['task'], 'checkpoint', checkpoint, )
            if not os.path.exists(cpt_path): os.makedirs(cpt_path)
            cpt_path = os.path.join(cpt_path, self.gv.logger.get_output_name(f'.{self.current_round}'))
            torch.save(cpt, cpt_path)

    def save_checkpoint(self):
        """Create the checkpoint to be saved

        Returns:
            cpt (dict): cpt dict
        """
        cpt = {
            'round': self.current_round,
            'learning_rate': self.learning_rate,
            'model_state_dict': self.model.state_dict(),
            'early_stop_option': {
                '_es_best_score': self.gv.logger._es_best_score,
                '_es_best_round': self.gv.logger._es_best_round,
                '_es_patience': self.gv.logger._es_patience,
            },
            'output': self.gv.logger.output,
            'time': self.gv.clock.current_time,
        }
        if self.option.get('save_optimal', False): cpt.update({'optimal_state': self.gv.logger._optimal_state})
        if hasattr(self.gv, 'simulator') and self.gv.simulator is not None:
            cpt.update({
                "simulator": self.gv.simulator.save_checkpoint()
            })
        return cpt

    def load_checkpoint(self, cpt):
        """Load state from the checkpoint

        Args:
            cpt (dict): cpt dict
        """
        md = cpt.get('model_state_dict', None)
        round = cpt.get('round', None)
        output = cpt.get('output', None)
        early_stop_option = cpt.get('early_stop_option', None)
        time = cpt.get('time', None)
        learning_rate = cpt.get('learning_rate', None)
        if md is not None: self.model.load_state_dict(md)
        if round is not None: self.current_round = round + 1
        if output is not None: self.gv.logger.output = output
        if time is not None: self.gv.clock.set_time(time)
        if learning_rate is not None: self.learning_rate = learning_rate
        if early_stop_option is not None:
            self.gv.logger._es_best_score = early_stop_option['_es_best_score']
            self.gv.logger._es_best_round = early_stop_option['_es_best_round']
            self.gv.logger._es_patience = early_stop_option['_es_patience']
        if self.option.get('save_optimal', False): self.gv.logger._optimal_state = cpt.get('optimal_state', None)
        if hasattr(self.gv, 'simulator') and self.gv.simulator is not None:
            self.gv.simulator.load_checkpoint(cpt.get('simulator', {}))

    def _load_checkpoint(self):
        checkpoint = self.option.get('load_checkpoint', '')
        if checkpoint != '':
            if os.path.exists(checkpoint) and os.path.isfile(checkpoint):
                cpt_name = os.path.split(checkpoint)[-1]
                cpt_round = cpt_name.split('.')[-1]
                cpt_path = checkpoint
            else:
                checkpoint = os.path.split(checkpoint)[-1]
                cpt_path = os.path.join(self.option['task'], 'checkpoint', checkpoint)
                if not os.path.exists(cpt_path): os.makedirs(cpt_path)
                cpt_name = self.gv.logger.get_output_name('')
                cpts = os.listdir(cpt_path)
                cpts = [p for p in cpts if cpt_name in p]
                if len(cpts) == 0: return False
                cpt_round = max([int(p.split('.')[-1]) for p in cpts])
                cpt_path = os.path.join(cpt_path, '.'.join([cpt_name, str(cpt_round)]))
            try:
                self.gv.logger.info(f'Loading checkpoint {cpt_name} at round {cpt_round}...')
                cpt = torch.load(cpt_path)
                self.load_checkpoint(cpt)
                return True
            except Exception as e:
                self.gv.logger.info(f"Failed to load checkpoint {cpt_path} due to {e}")
        return False

class BiRingProtocol(BasicProtocol):
    def __init__(self, option):
        super(BiRingProtocol, self).__init__(option)

    def init_topology(self):
        """Initialize the topology as a bi-direction ring"""
        for ck,c in enumerate(self.clients):
            c_clients = [self.clients[(ck+1)%self.num_clients], self.clients[(ck-1)%self.num_clients]]
            c.register_objects(c_clients, 'clients')
            c.num_clients = len(c_clients)
        self.topology = 'biring'

class RingProtocol(BasicProtocol):
    def __init__(self, option):
        super(RingProtocol, self).__init__(option)

    def init_topology(self):
        """Initialize the topology as a ring"""
        for ck,c in enumerate(self.clients):
            c_clients = [self.clients[(ck+1)%self.num_clients]]
            c.register_objects(c_clients, 'clients')
            c.num_clients = len(c_clients)
        self.topology = 'ring'

class BasicClient(fedbase.BasicServer):
    def __init__(self, option):
        super().__init__(option)
        self.clients = None
        self.num_clients = 0
        self.model = None
        self.state = {}
        self.pstate = {}
        self.nstate = {}
        self._train_loader = None
        self.data_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None
        # local calculator
        self.device = self.gv.apply_for_device()
        self.calculator = self.TaskCalculator(self.device, option['optimizer'])
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.batch_size = option['batch_size']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.clip_grad = option['clip_grad']
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        # system setting
        self._effective_num_steps = self.num_steps
        self._latency = 0
        # server
        # actions of different message type
        self.default_action = self.send_state
        self.register_cache_var('model')
        # actions of different message type
        self.actions = {0: self.send_state}

    def register_protocal(self, protocal=None):
        r"""
        Register the protocal to self.protocal
        """
        if protocal is not None: self.protocal = protocal

    def init_state(self):
        """Initialize local state"""
        # self.state will be shared with other clients and any direct change on self.state should not impact self.model
        self.state = copy.deepcopy({'model': self.model})

    def update_state(self):
        """Collect states from neighbors by communicating and return the locally updated state"""
        # sample clients
        selected_clients = self.sample()
        # request prior states from selected neighbors
        received_models = self.communicate(selected_clients)['model']
        # aggregate neighbors' states
        self.model = self.aggregate([self.model]+received_models)
        # local train
        self.train(self.model)
        return {'model': self.model}

    def aggregate(self, models: list, *args, **kwargs):
        """Aggregate models into a new model. Averaging models is as default.

        Args:
            models (list): a list of models

        Returns:
            agg_model (fmodule.FModule): the aggregated model
        """
        return fmodule._model_average(models)

    def pack(self, *args, **kwargs):
        return {}

    def send_state(self, *args, **kwargs):
        """Send the current read-only state to others.

        Returns:
            state (dict): the client's state. The receiver should not change write the state.
        """
        return self.state

    @fmodule.with_multi_gpus
    def train(self, model):
        r"""
        Standard local training procedure. Train the transmitted model with
        local training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return

    def get_batch_data(self):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size,
                                                                   num_workers=self.loader_num_workers,
                                                                   pin_memory=self.option['pin_memory'], drop_last=self.option.get('drop_last', False))
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self._train_loader)
            batch_data = next(self.data_loader)
        # clear local DataLoader when finishing local training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.

        Args:
            selected_clients (list of int): the clients to communicate with
            mtype (anytype): type of message
            asynchronous (bool): asynchronous communciation or synchronous communcation

        Returns:
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for client_id in communicate_clients:
            received_package_buffer[client_id] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            clients_for_iterate = communicate_clients
            for client_id in clients_for_iterate:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            self.model = self.model.to(torch.device('cpu'))
            try:
                import ray
            except:
                ray = None
            paratype = self.option.get('parallel_type', None)
            if paratype=='r' and ray is not None:
                @ray.remote(num_gpus=1.0/len(self.gv.dev_list))
                def wrap_communicate_with(server, client_id, package):
                    return server.communicate_with(client_id, package)
                for client_id in communicate_clients:
                    server_pkg = self.pack(client_id, mtype)
                    server_pkg['__mtype__'] = mtype
                    self.clients[client_id].update_device(self.gv.apply_for_device())
                    res_ref = wrap_communicate_with.remote(self, self.clients[client_id].id, server_pkg)
                    packages_received_from_clients.append(res_ref)
                ready_refs, remaining_refs = ray.wait(packages_received_from_clients, num_returns=len(communicate_clients), timeout=None)
                packages_received_from_clients = ray.get(ready_refs)
            else:
                # computing in parallel with torch.multiprocessing
                if paratype=='t':
                    pool = dmp.Pool(self.num_parallels)
                else:
                    pool = mp.Pool(self.num_parallels)
                for client_id in communicate_clients:
                    server_pkg = self.pack(client_id, mtype)
                    server_pkg['__mtype__'] = mtype
                    self.clients[client_id].update_device(self.gv.apply_for_device())
                    args = (self.clients[client_id].id, server_pkg)
                    packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
                pool.close()
                pool.join()
                packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))

            self.model = self.model.to(self.device)
            for pkg in packages_received_from_clients:
                for k, v in pkg.items():
                    if hasattr(v, 'to'):
                        try:
                            pkg[k] = v.to(self.device)
                        except:
                            continue
        for i, client_id in enumerate(communicate_clients): received_package_buffer[client_id] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

Client = BasicClient
Protocol = BasicProtocol