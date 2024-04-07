class SharedVars():

    def __init__(self):

        # Define visualization variables

        self.PAGESIZE = (10.5, 8)
        self.TITLE_FONTSIZE = 16
        self.SUBTITLE_FONTSIZE = 12
        self.TEXT_FONTSIZE = 10
        self.PDF_DPI = 600
        # self.VIS_PERCENTILE_MAX = 99.9

# Define instance of SharedVars class that will be accessible to (and editable by) other modules

SHARED_VARS = SharedVars()