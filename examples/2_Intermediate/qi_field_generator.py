import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from simsopt._core.optimizable import Optimizable


class BaseQIField():
    def __init__(self, Bmax, Bmin, nfp, iota, Dfunc, sfunc, phi=128, theta=64, neta=72):
        self.nfp = nfp
        self.Dfunc = Dfunc
        self.sfunc = sfunc

        # Below are properties
        self._Bmax = Bmax
        self._Bmin = Bmin
        self._iota = iota

        if isinstance(phi, int):
            self._nphi = phi
            self.phi = np.linspace(0, 2*np.pi/self.nfp, phi)
        else:
            self.phi = phi
            self._nphi = phi.size

        if isinstance(theta, int):
            self._ntheta = theta
            self.theta = np.linspace(0, 2*np.pi, theta)
        else:            
            self.theta = theta
            self._ntheta = theta.size

        
        self._neta = neta

        # setup inner grids
        self.need_to_update = True
 
    
    @property
    def Bmin(self):
        return self._Bmin
    def set_Bmin(self, new_value):
        self._Bmin = new_value
        self.need_to_update = True
        
    @property
    def Bmax(self):
        return self._Bmax
    def set_Bmax(self, new_value):
        self._Bmax = new_value
        self.need_to_update = True
        
    @property
    def iota(self):
        return self._iota
    def set_iota(self, new_value):
        self._iota = new_value
        self.need_to_update = True
        
    @property
    def nphi(self):
        return self._nphi
    def set_nphi(self, new_value):
        self._nphi = new_value
        self.need_to_update = True
        
    @property
    def ntheta(self):
        return self._ntheta
    def set_ntheta(self, new_value):
        self._ntheta = new_value
        self.need_to_update = True
        
    @property
    def neta(self):
        return self._neta
    def set_neta(self, new_value):
        self._neta = new_value
        self.need_to_update = True
        
    def get_tilde_coordinates(self, phi, theta):
        return phi*self.nfp, theta

    def phi_tilde_from_eta(self, eta, tt):
        eta = np.asarray(eta)
        tt = np.asarray(tt)

        # Set conditional evaluation
        cond = eta <= np.pi

        term1 = np.pi - self.sfunc(eta, tt + self._iota_tilde * self.Dfunc(eta)) - self.Dfunc(eta)
        
        eta2 = 2 * np.pi - eta
        term2 = np.pi + self.sfunc(eta2, -tt + self._iota_tilde * self.Dfunc(eta2)) + self.Dfunc(eta2)
    
        return np.where(cond, term1, term2)
    

    def get_eta(self, phi_tilde, theta_tilde):
        def root_func(eta, tt, pt):
            phi_tilde = self.phi_tilde_from_eta(eta, tt)
            return phi_tilde - pt
        
        # Make guess
        dist = (phi_tilde - self._phi_tilde_fw_grid)**2 + (theta_tilde - self._theta_fw_grid)**2
        ind = np.argmin(dist)
        eta_guess = self._eta_fw_grid.flatten()[ind]
    
        # Find root
        x0 = eta_guess
        ff = lambda x: root_func(x, theta_tilde, phi_tilde)
        sol = root(ff, x0, tol=1e-8)
    
        return sol.x


    def update(self):
        self._epsilon = (self.Bmax-self.Bmin) / (2*self.Bmin)
        
        self._theta_grid, self._phi_grid = np.meshgrid(self.theta, self.phi)
        
        # Get tilde (rotated) coordinates
        self._phi_tilde_grid, self._theta_tilde_grid = self.get_tilde_coordinates(self._phi_grid, self._theta_grid)
        self._iota_tilde = self.iota / self.nfp
        
        self._eta = np.linspace(0, 2*np.pi, self.neta)

        # prepare coordinate inversion - this is used to initialize root solve below
        self._eta_fw_grid, self._theta_fw_grid = np.meshgrid(self._eta, self.theta)
        self._phi_tilde_fw_grid = self.phi_tilde_from_eta(self._eta_fw_grid, self._theta_fw_grid)

        # Evaluate eta
        self._eta_grid = np.zeros(self._phi_tilde_grid.shape)
        for iphi in range(self.nphi):
            for itheta in range(self.ntheta):
                pt = self._phi_tilde_grid[iphi, itheta]
                tt = self._theta_tilde_grid[iphi, itheta]
                self._eta_grid[iphi, itheta] = self.get_eta(pt, tt)

        # Construct modB
        self._modB = self.Bmin * (1 + self._epsilon + self._epsilon * np.cos(self._eta_grid))
        self.need_to_update = False

    @property
    def modB(self):
        if self.need_to_update:
            self.update()

        return self._modB

    @property
    def eta_grid(self):
        if self.need_to_update:
            self.update()

        return self._eta_grid
        

    def get_modB_at_phi_theta(self, phi, theta):
        if self.need_to_update:
            self.update()
            
        phi_tilde, theta_tilde = self.get_tilde_coordinates(phi, theta)
        eta = self.get_eta(phi_tilde, theta_tilde)
        return self.Bmin * (1 + self._epsilon + self._epsilon * np.cos(eta))
        
        
    def plot_eta_contours_tilde(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self._phi_tilde_grid, self._theta_tilde_grid, self.eta_grid, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$\eta$')
        ax.set_xlabel(r'$\tilde\phi$')
        ax.set_ylabel(r'$\tilde\theta$')
        
    def plot_eta_contours(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self._phi_grid, self._theta_grid, self.eta_grid, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$\eta$')
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        
    def plot_modB_contours_tilde(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()

        CS = ax.contour(self._phi_tilde_grid, self._theta_tilde_grid, self.modB, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$|B|$')
        ax.set_xlabel(r'$\tilde\phi$')
        ax.set_ylabel(r'$\tilde\theta$')
        
    def plot_modB_contours(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self._phi_grid, self._theta_grid, self.modB, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$|B|$')
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        
    def plot_modB_contours_eta(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self._eta_grid, self._theta_tilde_grid, self.modB, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$|B|$')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$\tilde\theta$')


class OptimizableQIField(BaseQIField):
    def __init__(self, nfp, iota, Bmin, Bmax, Dparams, sparams, DN, sL, sN, phi=128, theta=64, neta=72):
        self._Bmax = Bmax
        self._Bmin = Bmin
        self._Dparams = Dparams
        self._sparams = sparams
        self.DN, self.sL, self.sN = DN, sL, sN

        self.build_s_and_D_functions()

        super().__init__(self.Bmax, self.Bmin, nfp, iota, self.Dfunc, self.sfunc, phi, theta, neta)
        self.need_to_update = False
        

    @property
    def Dparams(self):
        return self._Dparams
    @property
    def sparams(self):
        return self._sparams
    @property
    def Bmin(self):
        return self._Bmin
    @property
    def Bmax(self):
        return self._Bmax

    @classmethod
    def from_boozer_surface(cls, bsurf, DN, sL, sN, neta=72):
        if bsurf.need_to_run_code:
            raise ValueError("Need to run boozer surface first")

        bsurf.biotsavart.set_points(bsurf.surface.gamma().reshape((-1,3)))
        B = bsurf.biotsavart.B()
        modB = np.linalg.norm(B, axis=1)
        Bmin = np.min(modB)
        Bmax = np.max(modB)
        iota = bsurf.res['iota']
        nfp = bsurf.surface.nfp

        Dparams = np.zeros((DN,))
        sparams = np.zeros((sN*(sL+1)))
        
        return cls(nfp, iota, Bmin, Bmax, Dparams, sparams, DN, sL, sN,
            phi=bsurf.surface.quadpoints_phi * 2 * np.pi,
            theta=bsurf.surface.quadpoints_theta * 2 *np.pi,
            neta=neta
        )
            

    def set_dofs(self, new_values):
        if new_values.size!=2+self.DN+self.sN*(self.sL+1):
            raise ValueError("Invalid number of dofs")

        self._Bmax = new_values[0]
        self._Bmin = new_values[1]
        self._Dparams = new_values[2:self.DN+2]
        self._sparams = new_values[self.DN+2:]
        self.build_s_and_D_functions()    

    def get_concatenated_params(self):
        return np.concatenate([[self.Bmax, self.Bmin], self.Dparams, self.sparams])
    

    def build_s_and_D_functions(self):        
        def x_basis(x,l):
            return np.sin((l+1./2.)*x) #enforces 0 at origin

        def y_basis(y,n):
            return np.sin((n+1)*y)

        def Dfunc(x):
            first_term = np.pi - x
            second_term = sum([k*np.sin((i+1)*x) for i, k in enumerate(self._Dparams)])
            return first_term + second_term

        def sfunc(x,y):
            out = 0
            for l in range(self.sL+1):
                for n in range(self.sN):
                    out += self.get_sln(self.sparams, l, n) * x_basis(x,l) * y_basis(y, n)

            return out

        self.sfunc = sfunc
        self.Dfunc = Dfunc
        self.need_to_update = True

    def get_sln(self, sdofs, l, n):
        return self._sparams[l*self.sN + n]
        
    def set_sln(self, sdofs, l, n, mode):
        sdofs[l*self.sN + n] = mode

            




class BoozerSurfaceQI(BaseQIField, Optimizable):
    def __init__(self, bsurf, DN, sL, sN, phi=128, theta=64, neta=72, definition='goodman'):
        if bsurf.need_to_run_code:
            raise ValueError("Need to run boozer surface first")

        if definition not in ['goodman']:
            raise ValueError("Invalid definition")
        self.definition=definition
        
        self.boozer_surface = bsurf
        bsurf.biotsavart.set_points(bsurf.surface.gamma().reshape((-1,3)))
        B = bsurf.biotsavart.B()
        modB = np.linalg.norm(B, axis=1)
        Bmin = np.min(modB)
        Bmax = np.max(modB)
        iota = bsurf.res['iota']
        nfp = bsurf.surface.nfp
        
        self.DN, self.sL, self.sN = DN, sL, sN
        #dummy for initialization
        Dfunc = lambda x: np.pi-x
        sfunc = lambda x, y: 0
        BaseQIField.__init__(self, Bmax, Bmin, nfp, iota, Dfunc, sfunc, phi, theta, neta)

        
        ndofs = DN + sN*(sL+1) + 2
        x0 = np.zeros((ndofs,)) 
        x0[0] = Bmax
        x0[1] = Bmin
        Optimizable.__init__(self, x0=x0, depends_on=[bsurf], external_dof_setter=BoozerSurfaceQI.set_dofs, names=self.local_dof_names)

        self.build_s_and_D_functions()
        self.need_to_update = False
                    

    def set_dofs(self, new_values):
        if new_values.size!=self.full_x.size:
            raise ValueError("Invalid number of dofs")

        self.local_full_x = new_values
        self.build_s_and_D_functions()

    def get_dofs(self):
        return self.get_concatenated_params()

    @property
    def local_dof_names(self):
        names = ['Bmax', 'Bmin']
        for n in range(self.DN):
            names.append(f'Ds({n})')
        for l in range(self.sL+1):
            for n in range(self.sN):
                names.append(f'Ss({l},{n+1})')
        return names

    @property
    def Bmax(self):
        return self.local_full_x[0]
    @property
    def Bmin(self):
        return self.local_full_x[1]
    @property
    def Dparams(self):
        return self.local_full_x[2:2+self.DN]
    @property
    def sparams(self):
        return self.local_full_x[2+self.DN:]

    def set_Bmax(self, Bmax):
        self.local_full_x[0] = Bmax
    def set_Bmin(self, Bmin):
        self.local_full_x[1] = Bmin
    def set_Dparams(self, Dparams):
        self.local_full_x[2:self.DN+2] = Dparams
    def set_sparams(self, sparams):
        self.local_full_x[self.DN+2:] = sparams

    def build_s_and_D_functions(self):
        def x_basis(x,l):
            return np.sin((l+1./2.)*x) #enforces 0 at origin

        def y_basis(y,n):
            return np.sin((n+1)*y)

        def Dfunc(x):
            first_term = np.pi - x
            second_term = sum([k*np.sin((i+1)*x) for i, k in enumerate(self.Dparams)])
            return first_term + second_term

        def sfunc(x,y):
            out = 0
            for l in range(self.sL+1):
                for n in range(self.sN):
                    out += self.get_sln(self.sparams, l, n) * x_basis(x,l) * y_basis(y, n)

            return out

        self.sfunc = sfunc
        self.Dfunc = Dfunc
        self.need_to_update = True

    def get_sln(self, sdofs, l, n):
        return sdofs[l*(self.sN+1) + n]
        
    def set_sln(self, sdofs, l, n, mode):
        sdofs[l*(self.sN+1) + n] = mode


    # def J(self):
    #     self.bsurf.biotsavart.set_points(bsurf.surface.gamma().reshape((-1,3)))
    #     B = self.bsurf.biotsavart.B()
        
    #     nphi = bsurf.surface.quadpoints_phi.size
    #     ntheta = bsurf.surface.quadpoints_theta.size
    #     modB = np.linalg.norm(B, axis=1).reshape((nphi,ntheta))
        
    #     if self.definition=='goodman':
    #         objective = np.sqrt(np.sum((modB - OptQIField.modB)**2) / np.sum(modB**2))
    #     elif self.definition=='fourier':
    #         # penalize non-qs modes in eta-theta tilde space
    #         raise NotImplementedError("Not implemented")
    #     else:
    #         raise ValueError("Unknown definition")
    
    #     return objective
        
        
        
        














                
        
        