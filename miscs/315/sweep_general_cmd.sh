
proxychains python miscs/315/dynamic1.py --target ./datasets/celeba_hq_256/03437.jpg --cuda_id 0 --eta 1
proxychains python miscs/315/dynamic1.py --target ./datasets/celeba_hq_256/14255.jpg --cuda_id 1 --eta 1 
proxychains python miscs/315/dynamic1.py --target ./datasets/celeba_hq_256/21177.jpg --cuda_id 3 --eta 1 

proxychains python miscs/315/dynamic2.py --target ./datasets/celeba_hq_256/03437.jpg --cuda_id 4 --eta 1
proxychains python miscs/315/dynamic2.py --target ./datasets/celeba_hq_256/14255.jpg --cuda_id 5 --eta 1 
proxychains python miscs/315/dynamic2.py --target ./datasets/celeba_hq_256/21177.jpg --cuda_id 6 --eta 1 

proxychains python miscs/315/manifold.py --cuda_id 2 --eta 1 
proxychains python miscs/315/manifold.py --cuda_id 7 --eta 0.5
