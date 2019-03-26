import SchemDraw as schem
import SchemDraw.logic as l
import SchemDraw.elements as e

def create_ff(mainlabel='FF', label_l=None, label_t=None, label_r=None, label_b=None, clkSign=True):
  """create dynamically a flipflop

    All labels are lists with signal definition form top to bottom spread out evenly or from left to right.
    Labels can have Latex notation like '$\overline{Q}$'

    Args:
        label_l (list): List of IO label left ['D','E','clk']
        label_t (list): List of IO label left ['SET']
        label_r (list): List of IO label left ['Q', '$\overline{Q}$']
        label_b (list): List of IO label left ['CLR']
        noClkSign (bool): Defining if clk triangle should be drawn work only on left labels, (extents 'clk' to '  clk') 

    Returns:
        bool: The object

  """
  # Left IO
  if label_l:
    cnt = len(label_l)
    if cnt == 1:
      loc = [0.8]
    elif cnt == 2:
      loc = [0.8,0.2]
    elif cnt == 3:
      loc = [0.8,0.5,0.2]
    else:
      loc = np.linspace(1,0,cnt+2)[1:-1]
    if clkSign:
      for i in range(cnt):
        if label_l[i] == 'clk':
          loc_clk = loc[label_l.index('clk')]
          label_l[i] = '  clk'
        elif label_l[i] == 'CLK':
          loc_clk = loc[label_l.index('CLK')]
          label_l[i] = '  CLK'
    # Create IO's
    ff_iol  = {'cnt'    :cnt,
               'labels' :label_l,
               'loc'    :loc,
               'lblsize':15}
  else:
    ff_iol = None
  # Top IO
  if label_t:
    cnt = len(label_t)
    if cnt == 1:
      loc = [0.5]
    else:
      loc = np.linspace(1,0,cnt+2)[1:-1]
    # Create IO's
    ff_iot  = {'cnt'    :cnt,
               'labels' :label_t,
               'loc'    :loc,
               'lblsize':12}
  else:
    ff_iot = None

  # Right IO
  if label_r:
    cnt = len(label_r)
    if cnt == 1:
      loc = [0.8]
    elif cnt == 2:
      loc = [0.8,0.2]
    elif cnt == 3:
      loc = [0.8,0.5,0.2]
    else:
      loc = np.linspace(1,0,cnt+2)[1:-1]
    # Create IO's
    ff_ior  = {'cnt'    :cnt,
               'labels' :label_r,
               'loc'    :loc,
               'lblsize':15}
  else:
    ff_ior = None

  # Bottom IO
  if label_b:
    cnt = len(label_b)
    if cnt == 1:
      loc = [0.5]
    else:
      loc = np.linspace(1,0,cnt+2)[1:-1]
    # Create IO's
    ff_iob  = {'cnt'    :cnt,
               'labels' :label_b,
               'loc'    :loc,
               'lblsize':12}
  else:
    ff_iob = None


  if clkSign:
    if 'loc_clk' in locals():
      print(loc_clk)
      clkSign_path = [[[0,loc_clk-0.2],[0.3,loc_clk],[0,loc_clk+0.2]]]
    DFF_0 =  e.blackbox(d.unit, d.unit*1.5, linputs=ff_iol, rinputs=ff_ior, tinputs=ff_iot, binputs=ff_iob, mainlabel=mainlabel)
    DFF = { 'name'  : mainlabel,
            'base'  : DFF_0,
            'paths' : clkSign_path # Add clk triangle
            #'paths' : [[[0,0.7],[0.3,0.9],[0,1.1]]] # Add clk triangle
          }
    return DFF
  else:
    return e.blackbox(d.unit, d.unit*1.5, linputs=ff_iol, rinputs=ff_ior, tinputs=ff_iot, binputs=ff_iob, mainlabel=mainlabel)