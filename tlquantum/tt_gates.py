import tensorly as tl
tl.set_backend('pytorch')
from torch import rand, randn, cos, sin, complex64, exp, matrix_exp, sqrt, unsqueeze, pi
from torch.nn import Module, ModuleList, ParameterList, Parameter
from tensorly.tt_matrix import TTMatrix
from copy import deepcopy
from .tt_operators import identity
from .tt_precontraction import  qubits_contract, _get_contrsets, layers_contract
from .tt_sum import tt_matrix_sum
from math import factorial 

# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>
# Author: Jean Kossaifi <jkossaifi@nvidia.com>

# License: BSD 3 clause

class test_function(): 
    def __init__(self): 
        print('I am here!')
    
class Unitary(Module):
    """A unitary for all qubits in a TTCircuit, using tensor ring tensors
    with PyTorch Autograd support.
    Can be defined with arbitrary gates or used as a base-class for set circuit
    types.

    Parameters
    ----------
    gates : list of TT gate classes, each qubit in the unitary
            to be involved in one gate.
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    Unitary
    """
    def __init__(self, gates, nqubits, ncontraq, contrsets=None, dtype=complex64, device=None):
        super().__init__()
        if contrsets is None:
            contrsets = _get_contrsets(nqubits, ncontraq)
        self.nqubits, self.ncontraq, self.contrsets, self.dtype, self.device = nqubits, ncontraq, contrsets, dtype, device
        self._set_gates(gates)


    def _set_gates(self, gates):
        """Sets the gate class instances as a PyTorch ModuleList for Unitary.

        """
        self.gates = ModuleList(gates)


    def forward(self):
        """Prepares the tensors of Unitary for forward contraction by calling the gate instances'
        forward method and doing qubit-wise (horizonal) pre-contraction.

        Returns
        -------
        List of pre-contracted gate tensors for general forward pass.
        """
        return qubits_contract([gate.forward() for gate in self.gates], self.ncontraq, contrsets=self.contrsets)


class BinaryGatesUnitary(Unitary):
    """A Unitary sub-class that generates a layer of a single two-qubit gates accross
    all qubits in a TTCircuit.

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    q2gate : tuple of two gate instances, one for each qubit in gate.
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    BinaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, q2gate, parity, contrsets=None, random_initialization=True):
        dtype, device = q2gate[0].dtype, q2gate[0].device
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=dtype, device=device)
        self._set_gates(build_binary_gates_unitary(self.nqubits, q2gate, parity, dtype=dtype, random_initialization=random_initialization))


class UnaryGatesUnitary(Unitary):
    """A Unitary sub-class that generates a layer of unitary, single-qubit rotations.
    As simulation occurs in real-space, these rotations are about the Y-axis.

    Parameters
    ----------
    nqubits : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.

    Returns
    -------
    UnaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, axis='y', contrsets=None, dtype=complex64, device=None, params=None):
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=dtype, device=device)
        if params is None: params=[None]*nqubits
        if axis == 'y':
            self._set_gates([RotY(dtype=dtype, device=device, param=params[i]) for i in range(self.nqubits)])
        elif axis == 'x':
            self._set_gates([RotX(dtype=dtype, device=device, param=params[i]) for i in range(self.nqubits)])
        elif axis == 'z':
            self._set_gates([RotZ(dtype=dtype, device=device, param=params[i]) for i in range(self.nqubits)])
        else:
            self._set_gates([Rot(dtype=dtype, device=device, param=params[i]) for i in range(self.nqubits)])


def build_binary_gates_unitary(nqubits, q2gate, parity, random_initialization=True, dtype=complex64):
    """Generate a layer of two-qubit gates.

    Parameters
    ----------
    nqubits : int, number of qubits
    q2gate : tt-tensor, 2-core, 2-qubit gates to use in layer
    parity : int, if even, apply first q2gate core to even qubits, if odd, to odd qubits.

    Returns
    -------
    Layer of two-qubit gates as list of tt-tensors
    """
    def clone_gates(gate0, gate1, random_initialization):
        clone0, clone1 = deepcopy(gate0), deepcopy(gate1)
        if random_initialization:
            clone0.reinitialize(), clone1.reinitialize()
        return [clone0, clone1]

    q2gate0, q2gate1 = q2gate[0].type(dtype), q2gate[1].type(dtype)
    layer, device = [], q2gate0.device
    for i in range(nqubits//2 - 1):
        layer += clone_gates(q2gate0, q2gate1, random_initialization)
    if nqubits%2 == 0:
        temp = clone_gates(q2gate0, q2gate1, random_initialization)
        if parity%2 == 0:
            return layer+temp
        return [temp[1]]+layer+[temp[0]]
    temp = clone_gates(q2gate0, q2gate1, random_initialization)
    if parity%2 == 0:
        return layer+temp+[IDENTITY(dtype=dtype, device=device)]
    return [IDENTITY(dtype=dtype, device=device)]+layer+temp


class RotY(Module):
    """Qubit rotations about the Y-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotY
    """
    def __init__(self, dtype=complex64, device=None, param=None):
        super().__init__()
        if param is None:
            self.theta = Parameter(2*pi*rand(1, device=device))
        else: 
            self.theta= Parameter(tl.tensor(param, device=device))
        self.iden, self.epy = identity(dtype=dtype, device=self.theta.device), exp_pauli_y(dtype=dtype, device=self.theta.device)


    def forward(self):
        """Prepares the RotY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*cos(self.theta/2)+self.epy*sin(self.theta/2)


class RotX(Module):
    """Qubit rotations about the X-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotX
    """
    def __init__(self, dtype=complex64, device=None, param=None):
        super().__init__()
        if param is None:
            self.theta = Parameter(2*pi*rand(1, device=device))
        else: 
            self.theta= Parameter(tl.tensor(param, device=device))
        self.iden, self.epx = identity(dtype=dtype, device=self.theta.device), exp_pauli_x(dtype=dtype, device=self.theta.device)


    def forward(self):
        """Prepares the RotX gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*cos(self.theta/2)+self.epx*sin(self.theta/2)


class RotZ(Module):
    """Qubit rotations about the Z-axis with randomly initiated theta.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    RotZ
    """
    def __init__(self, dtype=complex64, device=None, param=None):
        super().__init__()
        if param is None:
            self.theta = Parameter(2*pi*rand(1, device=device))
        else: 
            self.theta= Parameter(tl.tensor(param, device=device))
        self.dtype, self.device = dtype, device
        self.iden, self.epz = identity(dtype=dtype, device=self.theta.device), exp_pauli_z(dtype=dtype, device=self.theta.device)

    def forward(self):
        """Prepares the RotZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*cos(self.theta/2)+self.epz*sin(self.theta/2)

class Rot(Module):
    """Qubit rotations about a random axis.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Rotation matrix
    """
    def __init__(self, dtype=complex64, device=None, param=None):
        super().__init__()
        if param is None:
            self.theta = Parameter(2*pi*rand(3, device=device))
        else: 
            self.theta= Parameter(tl.tensor(param, device=device))
        self.dtype, self.device =  dtype, device
        
    def forward(self):
        """
        Returns
        -------
        Gate tensor for general forward pass.
        """
        self.core = tl.zeros([1,2,2,1], dtype=self.dtype, device=self.device)
        self.core[0,0,0,0]= cos(self.theta[0]/2)
        self.core[0,1,1,0]= exp(1j*(self.theta[1]+self.theta[2]))*cos(self.theta[0]/2)
        self.core[0,0,1,0]= -exp(1j*self.theta[1])*sin(self.theta[0]/2)
        self.core[0,1,0,0]= exp(1j*self.theta[2])*sin(self.theta[0]/2)
        return self.core

class IDENTITY(Module):
    """Identity gate (does not change the state of the qubit on which it acts).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    IDENTITY
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        self.core, self.dtype, self.device = identity(dtype=dtype, device=device), dtype, device


    def forward(self):
        """Prepares the left qubit of the IDENTITY gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


def cnot(dtype=complex64, device=None):
    """Pair of CNOT class instances, one left (control) and one right (transformed).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    (CNOTL, CNOTR)
    """
    return CNOTL(dtype=dtype, device=device), CNOTR(dtype=dtype, device=device)


class CNOTL(Module):
    """Left (control-qubit) core of a CNOT gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CNOT gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype, device=device), dtype, device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CNOT gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


class CNOTR(Module):
    """Right (transformed qubit) core of a CNOT gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CNOT gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype, device=device), dtype, device
        core[0,0,0,0] = core[0,1,1,0] = 1.
        core[1,0,1,0] = core[1,1,0,0] = 1.
        self.core =  core


    def forward(self):
        """Prepares the right qubit of the CNOT gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


def cz(dtype=complex64, device=None):
    """Pair of CZ class instances, one left (control) and one right (transformed).

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    (CZL, CZR)
    """
    return CZL(dtype=dtype, device=device), CZR(dtype=dtype, device=device)


class CZL(Module):
    """Left (control-qubit) core of a CZ gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Left core of CZ gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        core, self.dtype, self.device = tl.zeros((1,2,2,2), dtype=dtype, device=device), dtype, device
        core[0,0,0,0] = core[0,1,1,1] = 1.
        self.core = core


    def forward(self):
        """Prepares the left qubit of the CZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


class CZR(Module):
    """Right (transformed qubit) core of a CZ gate.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    Right core of CZ gate.
    """
    def __init__(self, dtype=complex64, device=None):
        super().__init__()
        core, self.dtype, self.device = tl.zeros((2,2,2,1), dtype=dtype, device=device), dtype, device
        core[0,0,0,0] = core[0,1,1,0] = core[1,0,0,0]  = 1.
        core[1,1,1,0] = -1.
        self.core = core

    def forward(self):
        """Prepares the right qubit of the CZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of matrix representation.

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core


    def reinitialize(self):
        pass


def so4(state1, state2, dtype=complex64, device=None):
    """Pair of SO4 two-qubit rotation class instances, with rotations over
    different states.

    Parameters
    ----------
    state1 : int, the first of 4 quantum states to undergo the 2-qubit rotations
    state2 : int, the second of 4 quantum states to undergo the 2-qubit rotations
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4L, SO4R)
    """        
    R = SO4LR(state1, state2, 0, dtype=dtype, device=device)
    return R, SO4LR(state1, state2, 1, theta=R.theta, dtype=dtype, device=device)


class SO4LR(Module):
    """Left or right core of the two-qubit SO4 rotations gate.

    Parameters
    ----------
    state1 : int, the first of 4 quantum states to undergo the 2-qubit rotations
    state2 : int, the second of 4 quantum states to undergo the 2-qubit rotations
    position : int, if 0, then left core, if 1, then right core.
    device : string, device on which to run the computation.

    Returns
    -------
    if position == 0 --> SO4L
    if position == 1 --> SO4R
    """
    def __init__(self, state1, state2, position, theta=None, dtype=complex64, device=None):
        super().__init__()
        self.theta, self.position, self.dtype, self.device = Parameter(randn(1, device=device)), position, dtype, device
        if theta is not None:
            self.theta.data = theta.data
        ind1, ind2 = min(state1, state2), max(state1, state2)
        if (ind1, ind2) == (0,1):
            self.core_generator =  _so4_01
        elif (ind1, ind2) == (1,2):
            self.core_generator =  _so4_12
        elif (ind1, ind2) == (2,3):
            self.core_generator =  _so4_23
        else:
            raise IndexError('SO4 Rotation Gates have no state interaction pairs {}.\n'
                             'Valid state interactions pairs are (0,1), (1,2), and (2,3)'.format((state1, state2)))


    def forward(self):
        """Prepares the left or right qubit of the SO4 two-qubit rotation gate for forward contraction
        by calling the forward method and preparing the tt-factorized form of matrix representation.
        Update is based on theta (which is typically updated every epoch through backprop via Pytorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.core_generator(self.theta, dtype=self.dtype, device=self.device)[self.position]


    def reinitialize(self):
        self.theta.data = randn(1, device=self.device)


def _so4_01(theta, dtype=complex64, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along zeroth and first
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_01_L, SO4_01_R)
    """
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T01I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T23I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core2[0,1,0,0] = 1
    core2[0,0,1,0] = -1
    R23I = [core1*sin(theta), core2]
    return [*tt_matrix_sum(TTMatrix(T01I), tt_matrix_sum(TTMatrix(T23I), TTMatrix(R23I)))]


def _so4_12(theta, dtype=complex64, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along first and second
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_12_L, SO4_12_R)
    """
    core1, core2 = tl.zeros((1,2,2,2), dtype=dtype, device=device), tl.zeros((2,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core1[0,1,1,1] = core2[0,0,0,0] = core2[1,1,1,0] = 1
    T03I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,2), dtype=dtype, device=device), tl.zeros((2,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core1[0,0,0,1] = core2[0,0,0,0] = core2[1,1,1,0] = 1
    T12I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,2), dtype=dtype, device=device), tl.zeros((2,2,2,1), dtype=dtype, device=device)
    core1[0,1,0,0] = core1[0,0,1,1] = core2[0,0,1,0] = 1
    core2[1,1,0,0] = -1
    R12I = [core1*sin(theta), core2]
    return [*tt_matrix_sum(TTMatrix(T03I), tt_matrix_sum(TTMatrix(T12I), TTMatrix(R12I)))]


def _so4_23(theta, dtype=complex64, device=None):
    """Two-qubit SO4 gates in tt-tensor form with rotations along second and third
    qubit states.

    Parameters
    ----------
    theta : PyTorch parameter, angle about which to rotate qubit, optimizable with PyTorch Autograd
    device : string, device on which to run the computation.

    Returns
    -------
    (SO4_23_L, SO4_23_R)
    """
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,1,1,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T23I = [core1, core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core2[0,0,0,0] = core2[0,1,1,0] = 1
    T01I = [core1*cos(theta), core2]
    core1, core2 = tl.zeros((1,2,2,1), dtype=dtype, device=device), tl.zeros((1,2,2,1), dtype=dtype, device=device)
    core1[0,0,0,0] = core2[0,1,0,0] = 1
    core2[0,0,1,0] = -1
    R01I = [core1*sin(theta), core2]
    return [*tt_matrix_sum(TTMatrix(T23I), tt_matrix_sum(TTMatrix(T01I), TTMatrix(R01I)))]


def o4_phases(phases=None, dtype=complex64, device=None):
    """Pair of O4 phase rotations class instances. Each of four phases
    is imparted to each of the 4 states of O4.

    Parameters
    ----------
    phases : list of floats, the four phases to be imparted to the quantum states
    device : string, device on which to run the computation.

    Returns
    -------
    (O4L, O4R)
    """
    L = O4LR(0, phases=phases, dtype=dtype, device=device)
    phases = L.phases
    return [L, O4LR(1, phases=phases, dtype=dtype, device=device)]


class O4LR(Module):
    """Left and right core of the two-qubit O4 phase gate.

    Parameters
    ----------
    phases : list of floats, the four phases to be imparted to the quantum states
    device : string, device on which to run the computation.

    Returns
    -------
    Two-qubit unitary with general phase rotations for O4.
    """
    def __init__(self, position, phases=None, dtype=complex64, device=None):
        super().__init__()
        self.phases = [Parameter(randn(1, device=device)), Parameter(randn(1, device=device)), Parameter(randn(1, device=device)), Parameter(randn(1, device=device))]
        self.position, self.dtype, self.device = position, dtype, device
        if phases is not None:
            self.phases = [phases[0], phases[1], phases[2], phases[3]]


    def forward(self):
        """Prepares the left or right qubit of the SO4 two-qubit rotation gate for forward contraction
        by calling the forward method and preparing the tt-factorized form of matrix representation.
        Update is based on theta (which is typically updated every epoch through backprop via Pytorch Autograd).

        Returns
        -------
        Gate tensor for general forward pass.
        """
        core1, core2 = tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device), tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device)
        core1[0,0,0,0] = 1
        core2[0,0,0,0] = exp(1j*self.phases[0])
        core2[0,1,1,0] = exp(1j*self.phases[1])
        d0 = [core1, core2]
        core1, core2 = tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device), tl.zeros((1,2,2,1), dtype=self.dtype, device=self.device)
        core1[0,1,1,0] = 1
        core2[0,0,0,0] = exp(1j*self.phases[2])
        core2[0,1,1,0] = exp(1j*self.phases[3])
        d1 = [core1, core2]
        return tt_matrix_sum(d0, d1)[self.position]


    def reinitialize(self):
        for phase in self.phases:
            phase.data = randn(1, device=self.device)


def exp_pauli_y(dtype=complex64, device=None):
    """Matrix for sin(theta) component of Y-axis rotation in tt-tensor form.
    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) Y-rotation component.
    """
    return tl.tensor([[[[0],[-1]],[[1],[0]]]], dtype=dtype, device=device)


def exp_pauli_x(dtype=complex64, device=None):
    """Matrix for sin(theta) component of X-axis rotation in tt-tensor form.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) X-rotation component.
    """
    return tl.tensor([[[[0],[-1j]],[[-1j],[0]]]], dtype=dtype, device=device)

def exp_pauli_z(dtype=complex64, device=None):
    """Matrix for sin(theta) component of X-axis rotation in tt-tensor form.

    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) X-rotation component.
    """
    return tl.tensor([[[[-1j],[0]],[[0],[1j]]]], dtype=dtype, device=device)

def core_addition(c1, c2, end=0):
    if end==0: 
        return tl.concatenate((c1, c2), axis=3)
    elif end==-1: 
        return tl.concatenate((c1, c2), axis=0)
    else: 
        pc1 = tl.concatenate(
                (c1, tl.zeros((c2.shape[0], c1.shape[1], c1.shape[2], c1.shape[3]), device=c1.device))
                 , axis=0)
        pc2 = tl.concatenate(
                (tl.zeros((c1.shape[0], c1.shape[1], c1.shape[2], c2.shape[3]), device=c2.device), c2),
                axis=0)
        return tl.concatenate((pc1, pc2), axis=3)

def core_multiplication(f, core, i):
    return layers_contract([[f*core]]+[[core]*(i)], i+1)[0]

class Perceptron_U(Unitary):
    def __init__(self, nqubits, ncontraq, approx, dt=0.01, contrsets=None, device=None, Js=None, h=None):
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=complex64, device=device)
        self._set_gates([perceptron_U(approx=approx,dt=dt,device=device,Js=Js, h=h, end=0)]+\
        [perceptron_U(approx=approx,dt=dt,device=device,Js=Js, h=h, end=i) for i in range(1,nqubits-1)]+\
        [perceptron_U(approx=approx,dt=dt,device=device,Js=Js, h=h, end=-1)])

class perceptron_U(Module):        
    def __init__(self, approx=1, dt= 0.01, Js=None, h=None, device=None, end=0): 
        super().__init__()
        self.dt= dt
        self.end, self.approx, self.device= end, approx, device
        if end != -1:
            if Js is None: self.J = Parameter(2*rand(1, device=device)-1)
            else: self.J = Parameter(tl.tensor(Js[end], device=device))
        else: 
            if h is None: self.J = Parameter(2*rand(2, device=device)-1)
            else: self.J = Parameter(tl.tensor(h, device=device))
        
    def forward(self):
        self.core = IDENTITY(device=self.device).forward()
        if self.end==0:
            _core = tl.zeros((1,2,2,2),  device=self.device, dtype=complex64)
            _core[0,:,:,0] = tl.eye(2, device=self.device, dtype=complex64)
            _core[0,:,:,1] = self.J*tl.tensor([[1,0],[0,-1]], dtype=complex64, device=self.device)
            for i in range(self.approx):
                f = (-1j*self.dt)**(i+1)/factorial(i+1)
                self.core = core_addition(self.core,core_multiplication(f, _core, i), end=self.end)
        elif self.end==-1: 
            _core = tl.zeros((2,2,2,1), device=self.device, dtype=complex64)
            _core[1,:,:,0]=tl.tensor([[1,0],[0,-1]], dtype=complex64, device=self.device)
            _core[0,:,:,0]=self.J[1]*tl.tensor([[1,0],[0,-1]], dtype=complex64, device=self.device)
            _core[0,:,:,0]+=self.J[0]*tl.tensor([[0,1],[1,0]], dtype=complex64, device=self.device) 
            for i in range(self.approx):
                self.core = core_addition(self.core,core_multiplication(1., _core, i), end=self.end)
        else: 
            _core = tl.zeros((2,2,2,2), device=self.device, dtype=complex64)
            _core[0,:,:,0] = tl.eye(2, device=self.device, dtype=complex64)
            _core[1,:,:,1] = tl.eye(2, device=self.device, dtype=complex64)
            _core[0,:,:,1] = self.J*tl.tensor([[1,0],[0,-1]], dtype=complex64, device=self.device)
            for i in range(self.approx): 
                self.core = core_addition(self.core,core_multiplication(1., _core, i), end=self.end)
        return self.core

class Perceptron_WII(Unitary):
    def __init__(self, nqubits_total, ncontraq, dt=0.1, contrsets=None, device=None, Js=None, h=None):
        super().__init__([], nqubits_total, ncontraq, contrsets=contrsets, device=device)
        Win=nqubits_total-1
        if Js is None and h is None:
            gates =[star_wII(dt=dt,device=device, end=0)]+[star_wII(dt=dt,device=device, end=None) for i in range(1,Win)]
            gates+=[star_wII(dt=dt,device=device, end=1)]
        else:
            gates =[star_wII(dt=dt,device=device, end=0, j0=Js[0])]+[star_wII(dt=dt,device=device, end=None, j0=Js[i]) for i in range(1,Win)]
            gates+=[star_wII(dt=dt,device=device, end=1, h0=h)]
        self._set_gates(gates)

class star_wII(Module): 
    '''This class generates an approximation of the unitary evolution in MPO
    form for the star model. The star model consist of a number of input qubits
    interacting with a central output qubit via Ising type interaction: 
        H = Sum_{input}J_{input}S^z_{input}S^z_{output}+DS^z_{output}+OS^x_{output}
    This Hamiltonian can be efficiently written as an MPO of bond dimension 2. 
    The Approximation implemented here is W_II approx to exp(t H) from MPO 
    parts (A, B, C, D). The W_II approximation first appeared in `zaletel2015`.
    Here, the cores of the W_II approx can be computed algebraicaly, and we use that
    to hard-code them since tensorly lacks a matrix exponentiation function. 
    Parameter: 
    --------------------------------------------------------------------------
    dt (float): time of evolution. 
    device (string, None): device to run the simulation on
    end (int or None): this label tells us what core we need to prepare. 
    If end=0, the qubit is the first input qubit and so the core
    has size (1,2,2,2). 
    If end=None, the qubit is an input qubit but not the first one. 
    Thus, we can think of this as a bulk-qubit and the core is 
    of size (2,2,2,2). 
    If end=1, the qubit is the output qubit with core size (2,2,2,1). 
    Note that this core contains nontrivial terms coming from the 
    commutation of [Sx, Sz].'''
            
    def __init__(self, dt = 0.1, j0=None, h0=None, device=None, end=0): 
        '''Initiallize the module and create a core. Note that if the qubit
        is an input qubit, we only have one tunabble parameter J corresponding 
        to how the input qubit in question interacts with the output.'''
        super().__init__()
        self.end, self.device = end, device
        self.b  = tl.tensor([[0, 0], [1, 0]], dtype=complex64, device=device)
        self.I2 = tl.eye(2, dtype=complex64, device=device)
        self.I4 = tl.eye(4, dtype=complex64, device=device)
        self.Bc = tl.kron(self.I2, self.b)
        self.Br = tl.kron(self.b,self.I2)
        self.Brc = tl.kron(self.b,self.b)
        self.dt = tl.tensor([dt], device=device)
        
        if end==0:
            if j0 is None: self.J = Parameter(randn(1, device=device))
            else: self.J = Parameter(j0)
            self.core = tl.zeros((1,2,2,2), device=device, dtype=complex64)
        elif end is None:
            if j0 is None:self.J = Parameter(randn(1, device=device))
            else: self.J = Parameter(j0)
            self.core = tl.zeros((2,2,2,2), device=device, dtype=complex64)
        elif end==1: 
            if h0 is None:
                self.h=Parameter(randn(2, device=device)) #h[0]=O, h[1]=D
            else: self.h=Parameter(h0)
            self.core = tl.zeros((2,2,2,1), device=device, dtype=complex64)
        else: raise ValueError('End {} not supported'.format(self.end)) 
        
        self.prepare_core()
    
    def forward(self): 
        return self.core
        
    def prepare_core(self):
        tau = -1j*self.dt
        tc= (1+1j)*sqrt(self.dt/2)
        tb= (1-1j)*sqrt(self.dt/2)
        if self.end !=1:
            Dm=tl.zeros([2,2], dtype=complex64, device=self.device)
            Bm=Dm
            Cm=self.J*tl.tensor([[1,0],[0,-1]], dtype=complex64, device=self.device)
        else: 
            Dm=tl.tensor([[self.h[1],self.h[0]], [self.h[0],-self.h[1]]], dtype=complex64, device=self.device)
            Cm=tl.zeros([2,2], dtype=complex64, device=self.device)
            Bm=tl.tensor([[1,0],[0,-1]], dtype=complex64, device=self.device)
        
        w=matrix_exp(tl.kron(self.Brc,self.I2)+tb*tl.kron(self.Br, Bm)\
                     +tc*tl.kron(self.Bc, Cm)+tau*tl.kron(self.I4, Dm)).reshape([2]*6)
        
        if self.end==0: #(WD, WC)
            self.core[0,:,:,0]=w[0, 0, :, 0, 0, :]
            self.core[0,:,:,1]=w[0, 1, :, 0, 0, :]
        elif self.end is None: #((WD,WC),(WB,WA))
            self.core[0,:,:,0]=w[0, 0, :, 0, 0, :]
            self.core[0,:,:,1]=w[0, 1, :, 0, 0, :]
            self.core[1,:,:,1]=w[1, 1, :, 0, 0, :]
            self.core[1,:,:,0]=w[1, 0, :, 0, 0, :]
    
        elif self.end == 1: #((WD),(WB))
            self.core[0,:,:,0]= w[0, 0, :, 0, 0, :]
            self.core[1,:,:,0]= w[1, 0, :, 0, 0, :]
    
        else: raise ValueError('End {} not supported'.format(self.end)) 
        return