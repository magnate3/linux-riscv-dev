###############################################################################
 # A Data Plane native PPV PIE Active Queue Management Scheme using P4 on a Programmable Switching ASIC.
 # Karlstad University 2021.
 # Author: L. Dahlberg
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def SetPlotOptions(options, title = "", xlabel = "", ylabel = "", legend = False, line = False):
    """
    
    Set the options of the current plot.

    Args:
        options (dict): dict where each key contains a list with two entries, (color, legend text) 
        title (str, optional): Title of figure. Defaults to "".
        xlabel (str, optional): xlabel text. Defaults to "".
        ylabel (str, optional): ylabel text. Defaults to "".
        legend (bool, optional): Set True to display legend. Defaults to False.
        line (bool, optional): Set True to have legend display line type instead of color. Defaults to False.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if(legend):
        if(line):
            Elements = [Line2D([0], [0], linestyle=options[name][0], label='Line') for name in options.keys()]    
            plt.legend(Elements, [options[name][1] for name in options.keys()])
        else: 
            Elements = [Line2D([0], [0], color=options[name][0], lw=4, label='Line') for name in options.keys()]
            plt.legend(Elements, [options[name][1] for name in options.keys()])

class Plot():
    """
    
    Class used to Plot the information store within it using the library matplotlib. 

    Constructor args:
        x (list): x values to be plotted.
        y (list): y values to be plotted.
        option (str, optional): Optional parameter, usage differes between Plot types. Defaults to "".
        Type (str, optional): Optional parameter, usage differes between Plot types. Defaults to "".
        empty (bool, optional): Set True if the object is created as empty. Defaults to False.

    """
    notEmpty = True
    def __init__(self, x, y, option = "", Type = "", empty = False):
        self.x = x
        self.y = y
        self.option = option
        self.type = Type
        if(self.y == [] and not empty):
            self.notEmpty = False
    
    def plot(self, color = "red", x = "", y = "", line = "solid"):
        """
        Uses "plot" function in Matplotlib. 

        Args:
            color (str, optional): Color option. Defaults to "red".
            x (list, optional): Set to x value list. if not given use original x.
            y (list, optional): Set to y value list. if not given use original y.
            line (str, optional): Line option. Defaults to "solid".

        Returns:
            bool: True if non-empty.
        """
        if(self.notEmpty):
            if(y != "" and x != ""):
                plt.plot(x, y, color=color, linestyle = line)    
            else:
                plt.plot(self.x, self.y, color=color, linestyle = line)
            return True
        return False
    
    def scatter(self, color = "red", x = "", y = ""):
        """
        Uses "scatter" function in Matplotlib. 

        Args:
            color (str, optional): Color option. Defaults to "red".
            x (list, optional): Set to x value list. if not given use original x.
            y (list, optional): Set to y value list. if not given use original y.

        Returns:
            bool: True if non-empty.
        """
        if(self.notEmpty):
            if(y != "" and x != ""):
                plt.scatter(x, y, color=color)    
            else:
                plt.scatter(self.x, self.y, color=color)
            return True
        return False
        
    def hist(self, color = "red"):
        """
        
        Uses "hist" function in Matplotlib.

        Args:
            color (str, optional): Color option. Defaults to "red".

        Returns:
            bool: True if non-empty.
        """
        if(self.notEmpty):
            plt.hist(self.y, color=color)
            return True
        return False

    def bar(self, label, color = "red", correction = 0, edgecolor = "black", width = 0.4, bottom = 0):
        """
        Uses "bar" function in Matplotlib.

        Args:
            label (str): Label option.
            color (str, optional): Color option. Defaults to "red".
            correction (int, optional): Specifies the spacing between bars. Defaults to 0.
            edgecolor (str, optional): Edge color option. Defaults to "black".
            width (float, optional): Width of the bars. Defaults to 0.4.
            bottom (int, optional): bottom placement of bars. Defaults to 0.

        Returns:
            bool: True if non-empty.
        """
        if(self.notEmpty):
            average = sum(self.y)/len(self.y)
            plt.bar(int(label) + correction*width, average, color = color, width = width, edgecolor = edgecolor, linewidth = 3, bottom = bottom)
            return average
        return False

    def box(self,color = "red", labels = "", positions  = ""):
        """
        
        Uses "boxplot" function in Matplotlib.

        Args:
            color (str, optional): Color option. Defaults to "red".
            labels (str, optional): Label option . Defaults to "".
            positions (str, optional): Position option. Defaults to "".

        Returns:
            bool: True if non-empty.
        """
        if(self.notEmpty):
            p = plt.boxplot(self.y, labels = [labels], positions = [positions],  patch_artist=True)
            if(color != ""):
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(p[element], color=color)
                for patch in p['boxes']:
                    patch.set(facecolor=color)
                for flier in p['fliers']:
                    flier.set(markeredgecolor=color)
            return True
        return False

    def average(self, color = "red"):
        """
        
        Plots a single line at the average y-value using the "plot" function in Matplotlib.

        Args:
            color (str, optional): Color option. Defaults to "red".

        Returns:
            bool: True if non-empty.
        """
        if(self.notEmpty):
            average = sum(self.y)/len(self.y)
            plt.plot([self.x[0], self.x[-1]], [average, average], color = color)
            return True
        return False
