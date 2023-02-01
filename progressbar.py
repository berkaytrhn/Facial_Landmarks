from tqdm import tqdm
from time import sleep




array = [i for i in range(10)]


for k in tqdm(array, leave=False, ascii=True):

    inner_pbar = tqdm(total=100, leave=False, ascii=True)
    counter=0
    while True:
        if counter == 100:
            break
        sleep(0.0001)
        inner_pbar.update(1)
        counter+=1
    
    inner_pbar.close()

    