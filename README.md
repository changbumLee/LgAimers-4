### 코드
-----
        if canUseGPU
            executionEnvironment = "gpu";
            numberOfGPUs = gpuDeviceCount;
            pool = parpool(numberOfGPUs);
        else
            executionEnvironment = "cpu";
        end


-

        dataDir = 'C:\Users\jsgat\Documents\matlabData\mmdetection\dataset';
        annotationFile = fullfile(dataDir, 'train.json');
        jsonData = jsondecode(fileread(annotationFile));


-

        imageFolder = fullfile(dataDir, "train_all");
        imds = imageDatastore(imageFolder);
        
-

        annotations = jsonData.annotations;
        images = jsonData.images;
        numImages = numel(images);
        categories = jsonData.categories;
        classNames = {jsonData.categories.name};

% 어노테이션 저장을 위한 cell 배열

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


-

        ds = combine(imds, blds);

-

        inputSize = [224 224 3]; % Faster R-CNN은 기본적으로 224x224 이미지를 사용합니다.
        numClasses = numel(classNames);

% 앵커 박스 정의 (필요한 경우 사용자 정의 앵커 박스를 사용)

        anchorBoxes = [32 32; 64 64; 128 128; 224 224];

% 미리 학습된 ResNet-50 모델을 사용하여 Faster R-CNN 레이어를 생성합니다.

        network = resnet50;
        featureLayer = 'activation_40_relu';
        lgraph = fasterRCNNLayers(inputSize, numClasses, anchorBoxes, network, featureLayer);

%학습을 위한 옵션

        options = trainingOptions('sgdm', ...
            'MaxEpochs', 10, ...
            'MiniBatchSize', 4, ...
            'InitialLearnRate', 1e-3, ...
            'CheckpointPath', tempdir, ...
            'Verbose', true, ...
            'VerboseFrequency', 10, ...
            'Plots', 'training-progress');

%학습

        [detector, info] = trainFasterRCNNObjectDetector(ds, lgraph, options);

% 학습된 모델 저장 및 로드

        save('trainedFasterRCNN.mat', 'detector');
        load('trainedFasterRCNN.mat', 'detector');

%이미지 탐지를 할 이미지 폴더 경로

        testFolder = fullfile(dataDir, "test");

% 테스트 이미지에 대해 객체 탐지 수행

        testImages = imageDatastore(testFolder, 'FileExtensions', {'.jpg', '.png', '.jpeg'});
        while hasdata(testImages)
            I = read(testImages);
            [bboxes, scores, labels] = detect(detector, I, 'Threshold', 0.95);
            I = insertObjectAnnotation(I, 'rectangle', bboxes, labels);
            imshow(I);
            pause; % 다음 이미지를 보기 위해 일시 정지
        end
