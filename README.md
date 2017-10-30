# Darknet_mod

  
       
 # Для darknet_new 
	./darknet yolo train cfg/yolo.cfg extraction.conv.weights
	запускаем обучение. 
	В папке imgs - фото и их описания в формате Pascal VOC. 
	в файле train.txt пути к описаниям. 

	Если есть фото и описания, но нет train.txt то запускаем VKmake_train.py

# Следующие команды для /working_net. 

	./darknet yolo demo_cam  cfg/yolo.cfg backup/yolo_final_v2.weights -c 0 -thresh 0.45
	для того чтобы запустить распознавание на камере. -c 0 это номер камеры. -thresh 0.45 это порог от 0 до 1. Чем ниже порог тем больше ложных срабатываний. 


	./darknet yolo demo_vid  cfg/yolo.cfg backup/yolo_final_v2.weights 7.MP4 -thresh 0.45
	для того чтобы распознать на ранее записанном видео. 


	для RTSP 
	./darknet yolo demo_cam  cfg/yolo.cfg backup/yolo_final_v2.weights -c -2 -thresh 0.45
	и надо заглянуть в функцию demo_vid в /src/yolo_kenels.cu, а также в /src/yolo.c и потом вернуться в корень и сделать make. 

	ЛУЧШАЯ производительность. 
	./darknet yolo demo_cam  cfg/yolo-tiny.cfg backup/yolo_final_v2.weights -c -2 -thresh 0.45


