def ma(arr,window):
  res=[]
  for i in range(len(arr)):
    if i<window-1:
      res.append(None)
    else:
      res.append(sum(arr[i-window+1:i+1])/ window) 
  return res

def expo_ma(alpha,arr):
  res=arr[0]
  for i in range(1,len(arr)):
    res.append(res[-1]*alpha+(1-alpha)*arr[i])
  return res 
