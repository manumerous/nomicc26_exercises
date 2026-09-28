# water-net.mod   OOR2-AN-x-x
# AMPL coding: Sven Leyffer, University of Dundee, March 2000.

set nodes;					# nodes
set reservoirs within nodes;
set consumers within nodes;
set arcs within nodes cross nodes;		# arcs or connections

# ... node data
param demand{nodes};				# ... demand [m^3/sec]
param height{nodes};				# ... height over base [m]
param x{nodes};					# ... x-coordinate [m]
param y{nodes};					# ... y-coordinate [m]
param supply{nodes};				# ... supply [m^3/sec]
param wcost{nodes};				# ... [rp/m**3] cost of water at node i
param pcost{nodes};				# ... [rp/m**4] pumping cost at node i
param dist{(i,j) in arcs} 			# ... distance between nodes
      := sqrt( (x[i] - x[j])^2 + (y[i] - y[j])^2 );

# ... scalar parameters
param dpow := 5.33;		# ... power on diameter in pressure loss equation
param dmin := 0.15;		# ... minimum diameter of pipe
param dmax := 2.00;		# ... maximum diameter of pipe
param hloss := 1.03E-3;		# ... constant in the pressure loss equation
param dprc := 6.90E-2; 		# ... scale factor in the investment cost equation
param cpow := 1.29;		# ... power on diameter in the cost equation      
param r := 0.10;		# ... annual interest rate                               
param maxq := 2.00;		# ... bound on flow in pipe qp and qn                          
param davg := sqrt( dmin*dmax );# ... average diameter (geometric mean)
param rr        		# ... ratio of demand to supply
      := ( sum{i in consumers} demand[i] ) / ( sum{i in reservoirs} supply[i] );
param hl{i in nodes} := height[i] + if i in consumers then (7.5 + 5*demand[i]);

# ... variables
var qp{arcs} >= 0, <= maxq;		# ... flow on each arc - positive [m^3/sec]
var qn{arcs} >= 0, <= maxq;		# ... flow on each arc - negative [m^3/sec]
var d{arcs} >= dmin, <= dmax, := davg;	# ... pipe diameters [m]
var h{i in nodes} >= hl[i], := hl[i]+5;	# ... pressure at each node [m]
var s{ i in reservoirs} 		# ... supply at reservoir nodes
    >= 0, <= supply[i], := rr*supply[i];
var z{arcs} >= 0, <= 1, binary;		# ... binary variables

minimize cost: ( sum{i in reservoirs} s[i]*pcost[i]*( h[i] - height[i] )
                 + sum{i in reservoirs} s[i]*wcost[i] ) / r 
               + dprc * sum{(i,j) in arcs} ( dist[i,j] * d[i,j]^cpow );

subject to

   # ... flow conservation equation at each node
   cont{i in nodes}:  sum{(j,i) in arcs} (qp[j,i] - qn[j,i]) 
                    - sum{(i,j) in arcs} (qp[i,j] - qn[i,j]) 
                    + (if i in reservoirs then s[i]) = demand[i];

   # ... pressure loss on each arc (assumes qpow = 2)
   loss{(i,j) in arcs}: (h[i] - h[j])
                        = (hloss) * dist[i,j] * (qp[i,j]^2 - qn[i,j]^2) / (d[i,j]^dpow);

   # ... positive bounds on maximum flow
   qpup{(i,j) in arcs}: qp[i,j] <= maxq * z[i,j];
   qnup{(i,j) in arcs}: qn[i,j] <= maxq * (1 - z[i,j]);


