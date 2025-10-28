import numpy as np
from jax import numpy as jnp
from jax import grad, vjp, jvp, jacobian
from jax import jit as jaxjit
import matplotlib.pyplot as plt
from scipy.optimize import root, least_squares, root_scalar, minimize
from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import derivative_dec, Derivative
from simsopt.objectives.utilities import forward_backward

__all__ = ['BaseQIField', 'BoozerSurfaceQI']

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

    @property
    def eta_grid(self):
        if self.need_to_update:
            self.update()
        return self._eta_grid
    @property
    def phi_grid(self):
        if self.need_to_update:
            self.update()
        return self._phi_grid
    @property
    def theta_grid(self):
        if self.need_to_update:
            self.update()
        return self._theta_grid
    @property
    def phi_tilde_grid(self):
        if self.need_to_update:
            self.update()
        return self._phi_tilde_grid
    @property
    def theta_tilde_grid(self):
        if self.need_to_update:
            self.update()
        return self._theta_tilde_grid    
    
    @property
    def iota_tilde(self):
        if self.need_to_update:
            self.update()
        return self._iota_tilde
        
    def get_tilde_coordinates(self, phi, theta):
        return phi*self.nfp, theta

    def phi_tilde_from_eta(self, eta, tt):
        eta = np.asarray(eta)
        tt = np.asarray(tt)

        # Set conditional evaluation
        cond = eta <= np.pi

        # Lower than pi value
        term1 = np.pi - self.sfunc(eta, tt + self._iota_tilde * self.Dfunc(eta)) - self.Dfunc(eta)

        # Larger than pi value
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
        
        CS = ax.contour(self.phi_tilde_grid, self.theta_tilde_grid, self.eta_grid, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$\eta$')
        ax.set_xlabel(r'$\tilde\phi$')
        ax.set_ylabel(r'$\tilde\theta$')
        
    def plot_eta_contours(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self.phi_grid, self.theta_grid, self.eta_grid, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$\eta$')
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        
    def plot_modB_contours_tilde(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()

        CS = ax.contour(self.phi_tilde_grid, self.theta_tilde_grid, self.modB, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$|B|$')
        ax.set_xlabel(r'$\tilde\phi$')
        ax.set_ylabel(r'$\tilde\theta$')
        
    def plot_modB_contours(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self.phi_grid, self.theta_grid, self.modB, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$|B|$')
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        
    def plot_modB_contours_eta(self, ax=None, ncontours=18):
        if ax is None:
            ax = plt.figure().add_subplot()
        
        CS = ax.contour(self.eta_grid, self.theta_tilde_grid, self.modB, ncontours)
        cbar = plt.gcf().colorbar(CS)
        cbar.set_label(r'$|B|$')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(r'$\tilde\theta$')

    def plot_well(self, theta_start=0, npts=128, ax=None):
        phi_plot = np.linspace(0, 2*np.pi/self.nfp, npts)
        theta_plot = np.mod(theta_start + self.iota * phi_plot, 2*np.pi)

        modB = np.array([self.get_modB_at_phi_theta(p,t) for p, t in zip(phi_plot, theta_plot)])
        
        if ax is None:
            ax = plt.figure().add_subplot()

        ax.plot(phi_plot, modB)

    def get_bounce_distance(self, modB_target, theta_start=0):
        # First find minimum of well
        def func(phi):
            return self.get_modB_at_phi_theta(phi, theta_start+self.iota*phi)

        x0 = np.pi/self.nfp
        res = minimize(func, x0=x0)
        phi_min_well = res.x
        
        
        def func(phi, modB):
            return self.get_modB_at_phi_theta(phi, np.mod(theta_start+self.iota*phi, 2*np.pi)) - modB

        
        root_result = root_scalar(func, x0=0, x1=phi_min_well, args=(modB_target,), method='bisect', xtol=1e-6, bracket=[0, phi_min_well])
        x0 = root_result.root
        root_result = root_scalar(func, x0=phi_min_well, x1=2*np.pi/self.nfp, args=(modB_target,), method='bisect', xtol=1e-6, bracket=[phi_min_well, 2*np.pi/self.nfp])
        x1 = root_result.root

        return x1-x0

    def plot_bounce_distances(self, ax=None, nmodB=10, ntheta=32):
        if ax is None:
            ax = plt.figure().add_subplot()

        # define colormap
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=self.Bmin, vmax=self.Bmax)

        dmodB = self.Bmax-self.Bmin
        for modB in np.linspace(self.Bmin+dmodB/200,self.Bmax-dmodB/200,nmodB):
            theta_start = np.linspace(0,2*np.pi,ntheta)
            bounce_distance = []
            for t0 in theta_start:
                bounce_distance.append( self.get_bounce_distance( modB, theta_start=t0 ))
        
            plt.plot(theta_start,bounce_distance,color=cmap(norm(modB)))
        
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # needed for matplotlib < 3.7
        cbar = plt.gcf().colorbar(sm, ax=ax)
        cbar.set_label('|B| value')
        
        plt.xlabel('Field line label')
        plt.ylabel('Bounce distance')
        plt.title('Bounce distance vs θ for different |B| values')
        plt.tight_layout()
        

            

        

# ===============================================================================

def s_x_basis(x,l):
    return jnp.sin((l+1./2.)*x) #enforces 0 at origin

def s_y_basis(y,n):
    return jnp.sin((n+1)*y)

def Dfunc_pure(x, Dmodes):
    first_term = jnp.pi - x
    second_term = sum([k*jnp.sin((i+1)*x) for i, k in enumerate(Dmodes)])
    return first_term + second_term

def sfunc_pure(x,y,smodes,sL,sN):
    out = 0
    for l in range(sL+1):
        for n in range(sN):
            out += smodes[l*sN + n] * s_x_basis(x,l) * s_y_basis(y, n)

    return out



class BoozerSurfaceQI(BaseQIField, Optimizable):
    def __init__(self, bsurf, DN, sL, sN, neta=72, definition='goodman'):
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
        self.sfunc_jax = jaxjit(lambda x, y, sparams: sfunc_pure(x, y, sparams, self.sL, self.sN))
        self.Dfunc_jax = jaxjit(lambda x, Dparams: Dfunc_pure(x, Dparams))
        
        sfunc = lambda x,y: self.sfunc_jax(x,y,self.sparams)
        Dfunc = lambda x: self.Dfunc_jax(x,self.Dparams)

        phi = self.boozer_surface.surface.quadpoints_phi * 2 * np.pi
        theta = self.boozer_surface.surface.quadpoints_theta * 2 * np.pi
        
        BaseQIField.__init__(self, Bmax, Bmin, nfp, iota, Dfunc, sfunc, phi, theta, neta)

        # Jacobians are inefficient here - we don't need ds_ij/dy_kl, only ds_ij/dy_ij!
        # TODO: replace with vmaps and grad?
        self.d_sfunc_dx = lambda x,y: jacobian(self.sfunc_jax, argnums=0)(x, y, self.sparams)
        self.d_sfunc_dy = lambda x,y: jacobian(self.sfunc_jax, argnums=1)(x, y, self.sparams)
        self.d_sfunc_dsparams = lambda x,y: jacobian(self.sfunc_jax, argnums=2)(x, y, self.sparams)

        self.d_Dfunc_dx = lambda x: jacobian(self.Dfunc_jax, argnums=0)(x, self.Dparams)
        self.d_Dfunc_dDparams = lambda x: jacobian(self.Dfunc_jax, argnums=1)(x, self.Dparams)
        
        ndofs = DN + sN*(sL+1) + 2
        x0 = np.zeros((ndofs,)) 
        x0[0] = Bmax
        x0[1] = Bmin
        
        Optimizable.__init__(self, x0=x0, depends_on=[bsurf], names=self.local_dof_names)
        
        self.update()
        self.compute_objective_and_derivative()


        
    # def ds_x_basis_dx(x,l):
    #     return (l+1./2.)*np.cos((l+1./2.)*x)

    # def ds_y_basis_dy(y,n):
    #     return (n+1)*np.cos((n+1)*y)

    # def dDfunc(x, Dmodes):
    #     dfirst_term = -1.0
    #     dsecond_term = sum([k*(i+1)*np.cos((i+1)*x) for i, k in enumerate(Dmodes)])
    #     return dfirst_term + dsecond_term

    # def d_sfunc_dx(x,y,smodes):
    #     out = 0
    #     for l in range(self.sL+1):
    #         for n in range(self.sN):
    #             out += self.get_sln(smodes, l, n) * dx_basis(x,l) * y_basis(y, n)
                
    # def d_sfunc_dy(x,y,smodes):
    #     out = 0
    #     for l in range(self.sL+1):
    #         for n in range(self.sN):
    #             out += self.get_sln(smodes, l, n) * x_basis(x,l) * dy_basis(y, n)

    def set_dofs(self, new_values):
        self.local_x = new_values
        self.need_to_update = True

    def get_dofs(self):
        return np.concatenate([[self.Bmax, self.Bmin], self.Dparams, self.sparams])
        
    @property
    def local_dof_names(self):
        names = ['Bmax', 'Bmin']
        for n in range(self.DN):
            names.append(f'Ds({n+1})')
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
        self.need_to_update = True
    def set_Bmin(self, Bmin):
        self.local_full_x[1] = Bmin
        self.need_to_update = True
    def set_Dparams(self, Dparams):
        self.local_full_x[2:self.DN+2] = Dparams
        self.need_to_update = True
    def set_sparams(self, sparams):
        self.local_full_x[self.DN+2:] = sparams
        self.need_to_update = True
    

    def J(self):    
        return self._J


    @derivative_dec
    def dJ(self):
        return self._dJ


    def compute_objective_and_derivative(self):
        self.update()
        nphi = self.boozer_surface.surface.quadpoints_phi.size
        ntheta = self.boozer_surface.surface.quadpoints_theta.size
        
        self.boozer_surface.biotsavart.set_points(
            self.boozer_surface.surface.gamma().reshape((-1,3)) 
        )
        B = self.boozer_surface.biotsavart.B().reshape((nphi,ntheta,3))
        modB = np.linalg.norm(B, axis=2)

        num = np.sum((modB - self.modB)**2)
        denum = np.sum(modB**2)
        J = np.sqrt( num / denum )
        
        # Get derivative of J w.r.t to components of B
        d_num_dmodB = 2*(modB-self.modB)
        d_denum_dmodB = 2*modB
        dJ_dmodB = 1./(2*J) * ( (d_num_dmodB * denum - num * d_denum_dmodB) / denum**2 )

        
        # We need dJ / dself.modB
        dJ_dselfB = - 1./J * (modB-self.modB) / denum 

        #      nphi x ntheta  * nphi x ntheta x 3
        dmodB_dB = B / modB[:,:,None]
        dJdB = dJ_dmodB[:,:,None] * dmodB_dB # size nphi x ntheta x 3
        
        # Get derivative of field w.r.t surface dofs
        dB_by_dX = self.boozer_surface.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_ds = self.boozer_surface.surface.dgamma_by_dcoeff()
        dB_ds = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_ds, optimize=True)

        # Get derivative of modB w.r.t surface dofs
        dmodB_ds = (B[:, :, 0, None] * dB_ds[:, :, 0, :] + B[:, :, 1, None] * dB_ds[:, :, 1, :] + B[:, :, 2, None] * dB_ds[:, :, 2, :])/modB[:, :, None] # This is d|B| / dx

        dj_ds = np.einsum('ij,ijs->s', dJ_dmodB, dmodB_ds) # size ndof_surf

        # Use adjoint method to map this to coils dofs
        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = self.boozer_surface.res['vjp']
    
        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0], dtype=float)
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)

        self._J = J

        derivative_wrt_coils = self.boozer_surface.biotsavart.B_vjp(dJdB.reshape((-1,3))) - adj_times_dg_dcoil
        derivative_wrt_local_dofs = Derivative({
            self: np.einsum('ij,ijl->l' ,dJ_dselfB, self.d_modB_dparams()) #ok
        })
        self._dJ = derivative_wrt_coils + derivative_wrt_local_dofs
        

    def find_nearest_qi_field(self):
        stats = {'nfev': -1, 'njac': -1}
        
        def func_ls(dofs):
            stats['nfev'] += 1
        
            self.boozer_surface.biotsavart.set_points(self.boozer_surface.surface.gamma().reshape((-1,3)))
            B = self.boozer_surface.biotsavart.B()
            nphi = self.boozer_surface.surface.quadpoints_phi.size
            ntheta = self.boozer_surface.surface.quadpoints_theta.size
            modB = np.linalg.norm(B, axis=1).reshape((nphi, ntheta))
        
            self.set_dofs(dofs)
            residuals = (modB - self.modB).ravel() / np.sqrt(np.mean(modB**2))
        
            # print every ~10 evaluations (approximate iteration tracking)
            #if stats['nfev'] % 10 == 0:
            mean_resid = np.mean(residuals**2)
            print(f"nfev = {stats['nfev']:4d} :: mean residual = {mean_resid:.6E}")
        
            return residuals

        def jac(dofs):
            stats['njac'] += 1            
            
            self.boozer_surface.biotsavart.set_points(self.boozer_surface.surface.gamma().reshape((-1,3)))
            B = self.boozer_surface.biotsavart.B()
            nphi = self.boozer_surface.surface.quadpoints_phi.size
            ntheta = self.boozer_surface.surface.quadpoints_theta.size
            modB = np.linalg.norm(B, axis=1).reshape((nphi, ntheta))

            
            self.set_dofs(dofs)
            ndofs = dofs.size
            dmodB_dp = self.d_modB_dparams().reshape((-1, ndofs))
            
            jacobian = -dmodB_dp / np.sqrt(np.mean(modB**2))

            return jacobian
            
            
        
        x0 = self.get_dofs()
        res = least_squares(func_ls, x0, jac=jac, method='lm')  
        

    def d_modB_dparams(self):
        """
        Evaluate the derivatives of modB w.r.t {Bmax, Bmin, D-modes, s-modes}.
        """
        # Create empty array to store derivatives value. Has the size (nphi, ntheta, ndofs)
        ndofs = self.get_dofs().size
        grad = np.zeros((self.nphi,self.ntheta,ndofs))
        
        # Derivatives w.r.t Bmax, Bmin
        d_eps_d_Bmax = 1./(2*self.Bmin)
        d_modB_dBmax = self.Bmin * d_eps_d_Bmax * (1 + np.cos(self._eta_grid))
        grad[:,:,0] = d_modB_dBmax
        
        d_eps_d_Bmin = - self.Bmax/(2*self.Bmin**2)
        epsilon = (self.Bmax-self.Bmin)/(2*self.Bmin)
        d_modB_dBmin = (1 + epsilon * (1 + np.cos(self.eta_grid))) + self.Bmin * d_eps_d_Bmin * (1 + np.cos(self._eta_grid))
        grad[:,:,1] = d_modB_dBmin

        # Derivatives w.r.t Dparams and sparams
        # Set conditional evaluation - phi_tilde as a function of eta and theta_tilde is defined by piece.
        cond = self._eta_grid <= np.pi
        
        sgn = np.where(cond, -1.0, 1.0)
        sx = np.where(cond, self.eta_grid, 2*np.pi - self.eta_grid)
        Darg = np.where(cond, self.eta_grid, 2*np.pi - self.eta_grid)
        sy = - sgn * self.theta_tilde_grid + self.iota_tilde * self.Dfunc(Darg)

        # Now, this is not very efficient - the function d_sfunc_dx takes the derivatives of 
        # B_ij with respect to phi_k and theta_l, but we only really need the diagonal elements,
        # i.e. derivatives w.r.t to phi_i and theta_j. There is probably a better way of doing this,
        # but for now we use an np.einsum
        denom = np.einsum('ijij->ij', self.d_sfunc_dx(sx, sy) + (1.0 + self.d_sfunc_dy(sx, sy) * self._iota_tilde) * self.d_Dfunc_dx(Darg))

        # Again, same inefficient trick for the numerator. These are the derivatives w.r.t the 
        # modes of the D function
        a = (1. + self.d_sfunc_dy(sx,sy)*self._iota_tilde)
        b = self.d_Dfunc_dDparams(Darg) 
        d_eta_d_Dparams = sgn[:,:,None] * np.einsum('ijij,ijn->ijn',a,b) / denom[:,:,None]
        grad[:,:,2:2+self.DN] = - self.Bmin * epsilon * np.sin(self._eta_grid[:,:,None]) * d_eta_d_Dparams

        # And these are the derivatives w.r.t the modes of the s function
        d_eta_d_sparams = sgn[:,:,None] * self.d_sfunc_dsparams(sx, sy) / denom[:,:,None]
        grad[:,:,2+self.DN:]  = - self.Bmin * epsilon * np.sin(self._eta_grid[:,:,None]) * d_eta_d_sparams
        
        return grad

        
        
        
        














                
        
        
