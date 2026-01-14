import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer as fea
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp

task = './my_task'
if __name__ == '__main__':
    # generate federated task (remark: if task already exists, this line will not work)
    flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=100), task_path=task)

    # running fedavg on the specified task
    runner = flgo.init(task, fedavg, {'gpu':[0,],'log_file':True, 'num_epochs':1})
    runner.run()

    # visualize the experimental result
    records = fea.load_records(task, 'fedavg', {'num_epochs':1})
    fea.Painter(records).create_figure(fea.Curve,{
                'args':{'x':'communication_round', 'y':'val_loss'},
                'obj_option':{'color':'r'},
                'fig_option':{'xlabel': 'communication round', 'ylabel':'val_loss', 'title':'fedavg on {}'.format(task)}
            }
    )
    table = fea.Table(records)
    table.add_column(fea.min_value, {'x':'val_loss'})
    table.print()
