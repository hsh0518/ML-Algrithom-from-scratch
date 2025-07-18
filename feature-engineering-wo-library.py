class FeatureEngineering:
    def __init__(self,data):
        self.data = data
    def  one_hot_encode(self, column) :
        vals = list(set(self.data.column))
        for val in vals:
            if self.data.column == val:
                 self.data[column+'_'+val] = 1
            else :  self.data[column+'_'+val] = 0 
        del self.data.column
    def normalization(self,column): 
        vals = self.data[column]
        minv=min(vals) 
        maxv=max(vals) 
        if max !=min:
            self.data[column + "_norm"] = [(x-minv)/(max-minv)  for x in vals]
    def standardize (self, column):
        vals = self.data[column] 
        meanv = sum(vals)/len(vals)
        variance = sum((x-meanv)**2 for x in vals)
        std_dev =  (variance)**0.5
        self.data['standardized_'+column] =[( x -meanv)/std_dev for x in vals]
    def lable_encode(self,column):
        hm = {}
        for i, val in enumerate(set(self.data[column])):  
            hm[val] = i
        self.data[column] = [hm[val] for val in self.data[column]]
    def fillmissing (self, column, strategy = 'mean'):
        vals = self.data.coloumn
        if strategy == 'mean':
            fillv = sum(vals)/len(vals)
        if strategy == 'median':
            sorted_vals = sorted(vals)
            half = len(vals)//2
            fillv = (sorted_vals[half] if len(vals)%2 ==1  else (sorted_vals[half]+sorted_vals[half+1])/2 )
    def interaction(self, col1,col2)
        newcol = f'{col1}*{col2}'
        self.data[newcol]=[self.data.col1[i]*self.data.col2[i] for i in range(len(self.data[col1]))]
    def bin(self, col,bins):
        vals = self.data.col
        minv,maxv = min(vals),max(val)
        bin_width = (maxv-minv)/bins
        self.data[col+'_binned'] = [int(x-minv/bin_width) if x!=maxv else bin-1 for x in vals]


            

