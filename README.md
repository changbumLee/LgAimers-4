### 사전 준비 사항 
- Computer Vision Toolbox
- Parallel Computing Toolbox
- Deep Learning Toolbox
- Deep Learing Toolbox Model for ResNet-50 Network
  
### 코드
-----

#### 1. 실행환경 설정
parallel Computing Toolbox를 사용하여 Gpu 연결합니다. GPU 수에 따라 병렬 풀을 생성합니다.

        if canUseGPU
            executionEnvironment = "gpu";
            numberOfGPUs = gpuDeviceCount;
            pool = parpool(numberOfGPUs);
        else
            executionEnvironment = "cpu";
        end

#### 2. 데이터 불러오기 및 전처리
COCO JSON 형식의 어노테이션 파일을 불러와서 JSON 데이터를 디코딩합니다.

        dataDir = 'C:\Users\jsgat\Documents\matlabData\mmdetection\dataset';
        annotationFile = fullfile(dataDir, 'train.json');
        jsonData = jsondecode(fileread(annotationFile));


#### 3. 이미지 저장소 생성
이미지가 저장된 폴더를 지정하고, 해당 폴더에서 이미지를 읽어오는 imageDatastore를 생성합니다.
        
        imageFolder = fullfile(dataDir, "train_all");
        imds = imageDatastore(imageFolder);
        
#### 4. 어노테이션 데이터 불러오기
JSON 데이터를 불러와 변수에 지정합니다.
        
        annotations = jsonData.annotations;
        images = jsonData.images;
        numImages = numel(images);
        categories = jsonData.categories;
        classNames = {jsonData.categories.name};

#### 5. 어노테이션 데이터 처리
어노테이션 저장을 위한 cell 배열 선언하여, JSON 데이터에서 이미지를 읽어와서 바운딩 박스와 레이블 데이터를 처리한 후, 이를 boxLabelDatastore로 저장합니다.
        
        bboxData = cell(numImages, 1);
        labelData = cell(numImages, 1);
        
        for i = 1:numImages
            imgId = images(i).id;
            imgAnnotations = annotations([annotations.image_id] == imgId);
            
            bbox = zeros(numel(imgAnnotations), 4);
            label = strings(numel(imgAnnotations), 1);
            
            for j = 1:numel(imgAnnotations)
                bbox(j, :) = imgAnnotations(j).bbox; % [x, y, width, height]
                categoryId = imgAnnotations(j).category_id;
                label(j) = categories([categories.id] == categoryId).name;
            end
        
            bboxData{i} = bbox;
            labelData{i} = label;
        end
        
        blds = boxLabelDatastore(table(bboxData, labelData));


#### 6. 데이터 저장소 결합
이미지 데이터 저장소(imds)와 바운딩 박스 레이블 저장소(blds)를 결합하여 학습에 사용할 데이터 저장소(ds)를 생성합니다.
        
        ds = combine(imds, blds);

#### 7. 앵커 박스 정의 (필요한 경우 사용자 정의 앵커 박스를 사용)
        
        inputSize = [224 224 3]; % Faster R-CNN은 기본적으로 224x224 이미지를 사용합니다.
        numClasses = numel(classNames);

        anchorBoxes = [32 32; 64 64; 128 128; 224 224];

#### 8.  모델 및 학습 설정
미리 학습된 ResNet-50 모델을 사용하여 Faster R-CNN 레이어를 생성합니다.
        
        network = resnet50;
        featureLayer = 'activation_40_relu';
        lgraph = fasterRCNNLayers(inputSize, numClasses, anchorBoxes, network, featureLayer);

#### 9. 학습을 위한 옵션
학습을 위한 옵션을 설정합니다. Stochastic Gradient Descent with Momentum(SGDM) 옵티마이저를 사용하고, 에포크 수, 미니 배치 크기, 학습률 등을 지정합니다.

        options = trainingOptions('sgdm', ...
            'MaxEpochs', 10, ...
            'MiniBatchSize', 4, ...
            'InitialLearnRate', 1e-3, ...
            'CheckpointPath', tempdir, ...
            'Verbose', true, ...
            'VerboseFrequency', 10, ...
            'Plots', 'training-progress');

#### 10. 모델 학습

        [detector, info] = trainFasterRCNNObjectDetector(ds, lgraph, options);

#### 11. 학습된 모델 저장 및 로드

        save('trainedFasterRCNN.mat', 'detector');
        load('trainedFasterRCNN.mat', 'detector');

#### 12. 이미지 탐지를 할 이미지 폴더 지정
        testFolder = fullfile(dataDir, "test");

#### 13. 테스트 이미지에서 객체 탐지 수행
테스트 이미지 폴더에서 이미지를 읽어와서 객체 탐지를 수행하고, 결과를 이미지에 표시합니다.

        testImages = imageDatastore(testFolder, 'FileExtensions', {'.jpg', '.png', '.jpeg'});
        while hasdata(testImages)
            I = read(testImages);
            [bboxes, scores, labels] = detect(detector, I, 'Threshold', 0.95);
            I = insertObjectAnnotation(I, 'rectangle', bboxes, labels);
            imshow(I);
            pause; % 다음 이미지를 보기 위해 일시 정지
        end
