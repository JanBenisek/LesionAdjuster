'''====================================================================
-------------------------- USER INTERFACE -----------------------------
===================================================================='''
from bokeh.plotting import figure, show, Column, Row
from bokeh.models import DataTable, TableColumn, PointDrawTool, PolyEditTool, ColumnDataSource, CustomJS, Legend, Toggle
from bokeh.models.widgets import  Button , Div
from bokeh import events

import numpy as np
import cv2

'''====================================================================
-------------------------------- RESULT -------------------------------
===================================================================='''

class LesionAdjuster():
    ''' 
    Plots GUI in bokeh to modify the predictions made by U-Net.
    
    Notes
    ------
    Image, forma and prediction have shape (864, 1232) (rows, cols).
    Those are scaled by 70%, so they fit on a bokeh figure on screen
    Additionaly, previous work is based on cv2, where y-axis have 0 
      in the upper left corner, so translation had to be done.
    '''
    
    def __init__(self, root_pth, output_file='GUI.html', 
                 _tools_to_show='box_zoom,pan,save,hover,reset,tap',
                 scale=0.7):
        self.output_file = output_file
        self._tools_to_show = _tools_to_show
        self.scale = scale
        self.root_pth = root_pth
        
        self.DIMS = (862, 604) #cols(y), rows(x)
        self.yaxis = 864
        
        pass
    
    def showGUI(self, pth_to_img, y_form, pred):
        ''' 
        Method builds the bokeh GUI
        
        Parameters
        ----------
        pth_to_img: path to ultrasound image
        y_form: true form of the lesion
        pred: predicted form the lesion
        '''
        
        ##############
        #Set up a figure
        ##############
        p = figure(x_range=(0, self.DIMS[0]), y_range=(0, self.DIMS[1]), 
                   tools=self._tools_to_show,
                   plot_width=self.DIMS[0], 
                   plot_height=self.DIMS[1],
                   toolbar_location="above")
        
        #Add image as background        
        p.image_url(url=[self.root_pth + pth_to_img], 
                    x=431, y=302, w=862, h=604, anchor="center")
        
        #Nicier plot
        self._makeShiny(plot=p)
        
        ##############
        #Add lines and plot them
        ##############
        src_true, src_pred = self._getData()
        self._plotLines(plot=p, src_true=src_true, src_pred=src_pred)
        
        ##############
        #Add table
        ##############
        table = self._addTable(src_pred=src_pred)
        
        ##############
        #Add polygons
        ##############
        true_pol, c_t = self._addLesionForm(form=y_form, color='red', plot=p)
        pred_pol, c_p = self._addLesionForm(form=pred, color='blue', plot=p)
        
        #Add toggles for polygons
        toggle_true = Toggle(label="Show true form", button_type="primary", 
                             active=True)
        toggle_true.js_link('active', true_pol, 'visible')
        toggle_true.js_link('active', c_t, 'visible') 
        
        toggle_pred = Toggle(label="Show predicted form", button_type="primary", 
                             active=True)
        toggle_pred.js_link('active', pred_pol, 'visible')
        toggle_true.js_link('active', c_p, 'visible') 
        
        ##############
        #Add download button
        ##############
        button_csv = Button(label="Download", button_type="primary")
        button_csv.callback = CustomJS(args=dict(source=src_pred),
                                       code=open(self.root_pth + "download.js").read())
        
        ##############
        #Add title div
        ##############
        div_title = Div(text="""<div> <b>LESION ADJUSTER</b> </div>""", 
                         align='center',
                         style={'font-size': '150%', 
                                'color':'#1f77b4'})
        ##############
        #Add description to the buttons
        ##############
        div_desc = Div(text="""<div> <b>CONTROLS</b> </div>""", 
                         align='center',
                         style={'font-size': '110%', 
                                'color':'#1f77b4'})
        
        ##############
        #Add Div to show euclidean distance and button to recalculate it
        ##############
        div_euclid = Div(text="""
                         <b>Diameter of predicted form is:</b> 334.80 <br>
                         <b>Diameter of true form is:</b> 368.64 <br>
                         <b>RMSE is:</b> 34.13
                         """, 
                         align='center',
                         style={'font-size': '100%'})
        
        p.js_on_event(events.MouseMove,      
                      CustomJS(args=dict(div=div_euclid, 
                                        source_data_pred=src_pred,
                                        source_data_true=src_true),
               code="""
               var data_p = source_data_pred.data;
               var data_t = source_data_true.data;
               
               var x_p = data_p['x']
               var y_p = data_p['y']
               
               var x_t = data_t['x']
               var y_t = data_t['y']
               
               var diam_p = 0
               var diam_t = 0
               var rmse = 0
               
               //Diameter of pred form
               diam_p = Math.sqrt(Math.pow((x_p[0]-x_p[1]),2) + Math.pow((y_p[0]-y_p[1]),2))
               
               //Diameter of true form
               diam_t = Math.sqrt(Math.pow((x_t[0]-x_t[1]),2) + Math.pow((y_t[0]-y_t[1]),2))
               
               //RMSE
               rmse = Math.sqrt(Math.pow(diam_p - diam_t,2)/1)
               
               //Result
               div.text = "<b>Diameter of predicted form is: </b>" + diam_p.toFixed(2) + "<br> <b>Diameter of true form is: </b>" + diam_t.toFixed(2) + " <br> <b>RMSE is: </b>" + rmse.toFixed(2);
               
               """))
 
        ##############
        #Show
        ##############
        show(Column(div_title,
             Row(Column(p, table), 
                 Column(div_desc, toggle_true, toggle_pred, button_csv,
                        div_euclid))))                
    
    '''----------------------------------
                  AUX METHODS
    ---------------------------------'''
    def _makeShiny(self, plot):
        ''' 
        Modifg visuals of the figure
        
        Parameters
        ----------
        plot: instance of bokeh figure
        
        Output
        -----
        modifies the plot
        '''
        #plot.xgrid.visible = False
        #plot.ygrid.visible = False
        #plot.xaxis.visible = False
        #plot.yaxis.visible = False

        plot.background_fill_color = 'white'
        
    def _getData(self):
        ''' 
        Method calculates positions of the two lines in each lesion
        
        Parameters
        ----------
        -
        
        Output
        -------
        returns column sources with coordinates for lines
        '''   
        def lineSize(x_coords, y_coords):
            x_sum = (x_coords[0] - x_coords[1])**2
            y_sum = (y_coords[0] - y_coords[1])**2
            return (np.sqrt(x_sum + y_sum))
        
        ##############
        #DATA for true
        ##############
        x_coords_t = [x*self.scale for x in [651, 462]]
        y_coords_t = [y*self.scale  for y in [self.yaxis-175, self.yaxis-667]]
        
        source_true = ColumnDataSource(data=dict(x=x_coords_t, 
                                                 y=y_coords_t, 
                                                 color=['red', 'red'],
                                                 desc=['true','true']))        
        
        ##############
        #DATA for pred
        ##############
        x_coords_p = [x*self.scale for x in [348, 823]]
        y_coords_p = [y*self.scale  for y in [self.yaxis-529, self.yaxis-473]]
        
        source_pred = ColumnDataSource(data=dict(x=x_coords_p, 
                                                 y=y_coords_p, 
                                                 color=['blue', 'blue'],
                                                 desc=['pred','pred']))       
        
        return source_true, source_pred
    
    def _plotLines(self, plot, src_true, src_pred):
        ''' 
        Method add lines to the plot and a clickable legend
        
        Parameter
        --------
        plot: instance of bokeh figure
        src_true: ColumnDataSource line coords for true lesion
        src_pred: ColumnDataSource line coords for pred lesion
        
        Output
        -----
        adds lines to the plot, no explicit output
        '''
        
        ##############
        #Add lines
        ##############
        renderer_true_s = plot.scatter(x='x', y='y', 
                                    source=src_true, 
                                    color='color', size=15,
                                    line_color='red', line_width=5)
        
        renderer_true_l = plot.line(x='x', y='y', 
                                 source=src_true, color='red',
                                 line_dash='dashed', line_width=10)
            
        renderer_pred_s = plot.scatter(x='x', y='y', 
                                    source=src_pred,
                                    color='color', size=15, 
                                    line_color='blue', line_width=5)
        
        renderer_pred_l = plot.line(x='x', y='y', 
                                 source=src_pred, color='blue',
                                 line_dash='dashed', line_width=10)
        
        draw_tool = PointDrawTool(renderers=[renderer_pred_s, renderer_pred_l], 
                          empty_value='black')
        
        plot.add_tools(draw_tool)
        plot.toolbar.active_tap = draw_tool
        
        ##############
        #Legend (click to hide)
        ##############
        legend = Legend(items=[
            ("True diameter"   , [renderer_true_l, renderer_true_s]),
            ("Pred diameter" , [renderer_pred_l, renderer_pred_s])], 
            location="center")
        
        plot.add_layout(legend, 'right')
        plot.legend.click_policy="hide"
        
    def _addTable(self, src_pred):
        ''' 
        Method adds table under the plot with position of the lines
        
        Parameters
        ----------
        src_pred: ColumnDataSource line coords for pred lesion
        
        Output
        ------
        Returns a table object to be added to the figure
        
        '''
        columns = [TableColumn(field='desc', title='description'),
                   TableColumn(field='color', title='color'),
                   TableColumn(field="x", title="X coordinates"),
                   TableColumn(field="y", title="Y coordinates")]

        table = DataTable(source=src_pred, columns=columns, 
                          editable=True, height=400)
        return table
    
    def _addLesionForm(self, form, color, plot):
        ''' 
        Method adds lesion form, either true or pred
        
        Parameters
        ---------
        plot: instance of bokeh figure
        form: true or predicted form of the lesion
        color: color of the lesion (red=true, blue=pred)
        
        Output
        ------
        Returns instance of a polygon
        '''
        #to uint
        form = np.uint8(form)
        
        #Find contours
        cnts = cv2.findContours(form, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]
        
        #Scale to correct xy coords
        xy_coords = cnts[0][:,0,:]
        xy_coords[:, 1] = (form.shape[0]-1)-xy_coords[:, 1] #first is number of rows, so y

        #Lists of x and y coords        
        xs_m = list(xy_coords[:,0])
        ys_m = list(xy_coords[:,1])
        
        #Scale
        xs_m = [i * self.scale for i in xs_m]
        ys_m = [i * self.scale for i in ys_m]
        
        ##############
        #Add pplygon
        ##############
        p1 = plot.patches([], [], fill_alpha=0.4)
        c1 = plot.circle([], [], size=10, color=color)
        poly = plot.patches(xs=[xs_m], 
                            ys=[ys_m],
                            fill_color=color,
                            line_color=color,
                            line_alpha=0.5,
                            fill_alpha=0.5)
        
        #Add edit tool (only for pred)
        if color == 'blue':
            edit_tool = PolyEditTool(renderers=[p1, poly], vertex_renderer=c1)
            plot.add_tools(edit_tool)
            plot.toolbar.active_drag = edit_tool           
      
        return poly, c1
    