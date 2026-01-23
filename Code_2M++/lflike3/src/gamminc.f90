!--------------------------------------------------
!module lflike

!ontains

!----------------------------------------------------------------------
    double precision function gamincc(a, x)
   
      ! generalized routine for complement of incomplete gamma function 
      ! (unnormalized),
      ! i.e., $ \int_x^\infty \exp(-t) t^{a-1} dt.
      !
      ! see Abramovitz and Stegun, 6.5.3 ( \Gamma(a,x) ) p. 260
      ! but generalized for any value of a, including a<0, using the 
      ! recurrence relation (repeated integration by parts)
      
      ! Fails if a<-1 and x=0
      !       or a=negative integer
   
      implicit none
      double precision,intent(in) :: a,x
      double precision :: c,e,fact,gamser,gln,gammcf
      integer :: i,n

      c = a
      fact = 1.d0
      gamincc = 0.d0
      
      if ( a.lt.0.d0 ) then
         n = int(-a+1)
         c = a + n
         e = exp(-x)
         do i = n, 1, -1
            fact = fact/(c-float(i))
            gamincc = gamincc - fact*x**(c-float(i))*e
         end do
      end if
      
      ! Now get the functions from Numerical Recipes.  
      ! We want the unnormalized form,
      ! hence the EXP(GLN) term
      
      if ( x.lt.c+1. ) then
         call gser(gamser, c, x, gln)
         gamincc = gamincc + fact*exp(gln)*(1.-gamser)
      else
         call gcf(gammcf, c, x, gln)
         gamincc = gamincc + fact*exp(gln)*gammcf
      endif
      
      return
      
    end function gamincc
    !---------------------------------------------------------------    
    SUBROUTINE gser(gamser,a,x,gln)
      IMPLICIT NONE

      INTEGER ITMAX
      DOUBLE PRECISION a,gamser,gln,x,EPS
      PARAMETER (ITMAX=100,EPS=3.d-16)
      !    USES gammln
      INTEGER n
      DOUBLE PRECISION ap,del,sum
      double precision gammln
      gln=gammln(a)
      if(x.le.0.d0)then
         if(x.lt.0.d0)stop 'x < 0 in gser'
         gamser=0.d0
         return
      endif
      ap=a
      sum=1.d0/a
      del=sum
      do 11 n=1,ITMAX
         ap=ap+1.d0
         del=del*x/ap
         sum=sum+del
         if(abs(del).lt.abs(sum)*EPS)goto 1
11    continue
      stop 'a too large, ITMAX too small in gser'
1     gamser=sum*exp(-x+a*log(x)-gln)
      return
    END SUBROUTINE gser    
!---------------------------------------------------
    SUBROUTINE gcf(gammcf,a,x,gln)
      IMPLICIT NONE

      INTEGER ITMAX
      DOUBLE PRECISION a,gammcf,gln,x,EPS,FPMIN
      PARAMETER (ITMAX=200,EPS=3.d-16,FPMIN=1.d-30)
      !    USES gammln
      INTEGER i
      DOUBLE PRECISION an,b,c,d,del,h
      double precision gammln
      gln=gammln(a)
      b=x+1.d0-a
      c=1.d0/FPMIN
      d=1.d0/b
      h=d
      do 11 i=1,ITMAX
        an=-i*(i-a)
        b=b+2.d0
        d=an*d+b
        if(abs(d).lt.FPMIN)d=FPMIN
        c=b+an/c
        if(abs(c).lt.FPMIN)c=FPMIN
        d=1.d0/d
        del=d*c
        h=h*del
        if(abs(del-1.d0).lt.EPS)goto 1
11    continue
      stop 'a too large, ITMAX too small in gcf'
1     gammcf=exp(-x+a*log(x)-gln)*h
      return
      END SUBROUTINE gcf
!---------------------------------------------------
      FUNCTION gammln(xx)

      IMPLICIT NONE

      DOUBLE PRECISION gammln,xx
      INTEGER j
      DOUBLE PRECISION ser,stp,tmp,x,y,cof(6)
      SAVE cof,stp

      DATA cof,stp/76.18009172947146d0,-86.50532032941677d0,&
           24.01409824083091d0,-1.231739572450155d0,&
           .1208650973866179d-2,-.5395239384953d-5,2.5066282746310005d0/
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      gammln=tmp+log(stp*ser/x)
      return
    END FUNCTION gammln 
    
!end module lflike

