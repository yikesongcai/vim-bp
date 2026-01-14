"""
VIM-BP Algorithm Module

This module provides the VIM-BP algorithm in a format compatible with FLGo's flgo.init().
It exports Server and Client classes that FLGo expects.
"""

from vim_bp.vim_server import VIMServer as Server
from vim_bp.vim_client import VIMClient as Client

# Algorithm name for FLGo logging
__name__ = 'vim_algorithm'
