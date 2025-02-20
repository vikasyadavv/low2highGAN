%% Load the Generator and Discriminator Model
load 'path to the generator network';
load 'path to the discriminator network';

% Load the dataSet, conatains data(without wavelength), W(wavelength),
% labels(if any), and name(name of the minearl)

load 'high resolution Raman dataset';
load 'low resolution Raman dataset';

% preprocess the high resolution dataset; Herein we divided the data into 70% training, 15% validation, and rest 15% testing. So, out of 400 Raman spectra, 340 were utilized for th training and validation purpose.
Wnew = W(1:862,:);
data_high_resolution = [aspirin_andor(1:862,1:340), ibuprofen_andor(1:862,1:340), pcm_andor(1:862,1:340), ranitidine_andor(1:862,1:340), mba_andor(1:862,1:340), ntp_andor(1:862,1:340)];

% normalize the high resolution Raman dataset in the range of 0 to 1.
data_high_resolution = normalize(data_high_resolution,'range');
labels_high_resolution = [linspace(1,1,340), linspace(2,2,340), linspace(3,3,340), linspace(4,4,340), linspace(5,5,340), linspace(6,6,340)];

% preprocess the low resolution dataset; Herein we divided the data into 70% training, 15% validation, and rest 15% testing. So, out of 400 Raman spectra, 340 were utilized for th training and validation purpose.
data_low_resolution = [aspirin_goya(1:862,1:340), ibuprofen_goya(1:862,1:340), pcm_goya(1:862,1:340), ranitidine_goya(1:862,1:340), mba_goya(1:862,1:340), ntp_goya(1:862,1:340)];

% normalize the low resolution Raman dataset in the range of 0 to 1.
data_low_resolution = normalize(data_low_resolution,'range');
labels_low_resolution = [linspace(1,1,340), linspace(2,2,340), linspace(3,3,340), linspace(4,4,340), linspace(5,5,340), linspace(6,6,340)];


% heldout data for the testing of the trained model
heldData = [aspirin_goya(1:862, 341:end), ibuprofen_goya(1:862, 341:end), pcm_goya(1:862, 341:end), ranitidine_goya(1:862, 341:end), mba_goya(1:862, 341:end), ntp_goya(1:862, 341:end)];

target = data_high_resolution;

% Arrange the data set into format "wavelength point * No of Raman spectra"; if already done ignore it. The data contains 862 wavelnegth points and a total of 340*6 = 2040 Raman spectra to trained the model from low resolution to high resolution and 340 Raman spectra to test the model.

sizeData = [862 1 1 2040];


options = p2p.trainingOptions();
options.InputChannels = 1;
options.InputSize = [862 1];
options.OutputChannels = 1;
options.ExecutionEnvironment = 'gpu';
options.MaxEpochs = 200;
options.MiniBatchSize = 1;
options.ResumeFrom = 'path of the directory to store the trained or the checkpoint model';

if (options.ExecutionEnvironment == "auto" && canUseGPU) || ...
        options.ExecutionEnvironment == "gpu"
    env = @gpuArray;
else
    env = @(x) x;
end

if ~isempty(options.CheckpointPath)
    % Make a subfolder for storing checkpoints
    timestamp = strcat("p2p-", datestr(now, 'yyyymmdd-HHMMSS'));
    checkpointSubDir = fullfile(options.CheckpointPath, timestamp);
    mkdir(checkpointSubDir)
end

combinedChannels = options.InputChannels + options.OutputChannels;



if isempty(options.ResumeFrom)
    g = net_1; % net_1 is the generator network which we have loaded above
    d = net_2; % net_2 is the discriminator network which we have loaded above

    gOptimiser = p2p.util.AdamOptimiser(options.GLearnRate, options.GBeta1, options.GBeta2);
    dOptimiser = p2p.util.AdamOptimiser(options.DLearnRate, options.DBeta1, options.DBeta2);

    iteration = 0;
    startEpoch = 1;
else
    data = load(options.ResumeFrom, 'p2pModel');
    g = data.p2pModel.g;
    d = data.p2pModel.d;
    gOptimiser = data.p2pModel.gOptimiser;
    dOptimiser = data.p2pModel.dOptimiser;

    iteration = gOptimiser.Iteration;
    startEpoch = "the number of epoch to start from";
end

S = single(reshape(data_low_resolution, sizeData)); % training and validation data
L = single(reshape(target, sizeData)); % labels
miniBatchSize = options.MiniBatchSize;
totIter =  floor(size(S, 4)/miniBatchSize);

if options.Plots == "training-progress"
    %examples = imageAndLabel.shuffle();
    %nExamples = 9;
    %examples.MiniBatchSize = nExamples;
    %data = examples.read();
    idx = randperm(size(S, 4));
    thisInput = S(:, :, :, idx);

    exampleInputs = dlarray(env(thisInput), 'SSCB');
    trainingPlot = p2p.vis.TrainingPlot(exampleInputs);
end


%% Training loop
for epoch = startEpoch:options.MaxEpochs

    % Reset and shuffle data
    idx = randperm(size(S, 4));
    S = S(:, :, :, idx);
    L = L(:, :, :, idx);

    for rowIndex = 1:totIter

        iteration = iteration + 1;

        idx = (rowIndex-1)*miniBatchSize+1:rowIndex*miniBatchSize;

        X = S(:, :, 1, idx);
        T = L(:, :, 1, idx);
        %Z = randn(1, 1, numLatentInputs, miniBatchSize, 'single');

        % Convert mini-batch of data to dlarray and specify dimension
        % labels 'SSCB' (spatial, spatial, channel, batch)
        inputImage = dlarray(X, 'SSCB');
        %thisTarget = dlarray(Z, 'SSCB');
        targetImage = dlarray(T, 'SSCB');



        [g, gLoss, d, dLoss, lossL1, ganLoss, ~] = ...
            dlfeval(@stepBoth, g, d, gOptimiser, dOptimiser, inputImage, targetImage, options);

        if mod(iteration, options.VerboseFrequency) == 0
            logArgs = {epoch, iteration,  ...
                gLoss, lossL1, ganLoss, dLoss};
            fprintf('epoch: %d, it: %d, G: %f (L1: %f, GAN: %f), D: %f\n', ...
                logArgs{:});
            if options.Plots == "training-progress"
                trainingPlot.update(logArgs{:}, g);
            end

            if mod(iteration,5000) == 0 || iteration == 1
                % Generate signals using held-out generator input
                dlXGeneratedValidation = predict(g, inputImage);
                dlXGeneratedValidation = squeeze(extractdata(gather(dlXGeneratedValidation)));

                % Display spectra of validation signals
                figure;
                %subplot(1,2,1);
                plot(dlXGeneratedValidation);
                % set(gca, 'XScale', 'log')
                legend('Generated Signal')
                title("Spectra of Generated Signals")
            end


        end
    end

    p2pModel = struct('g', g, 'd', d, 'gOptimiser', gOptimiser, 'dOptimiser', dOptimiser);
    if ~isempty(options.CheckpointPath)
        checkpointFilename = sprintf('p2p_checkpoint_%s_%04d.mat', datestr(now, 'YYYY-mm-DDTHH-MM-ss'), epoch);
        p2pModel = gather(p2pModel);
        save(fullfile(checkpointSubDir, checkpointFilename), 'p2pModel')
    end
end

function [g, gLoss, d, dLoss, lossL1, ganLoss, images] = stepBoth(g, d, gOpt, dOpt, input, target, options)

% generate the data
generated = tanh(g.forward(input));

%% D update
% Apply the discriminator
realPredictions = sigmoid(d.forward(...
    cat(3, target, input) ...
    ));
generatedPredictions = sigmoid(d.forward(...
    cat(3, generated, input)...
    ));

% calculate D losses
labels = ones(size(generatedPredictions), 'single');
% crossentropy divides by nBatch, so we need to divide further
dLoss = options.DRelLearnRate*(crossentropy(realPredictions, labels)/numel(generatedPredictions(:,:,1,1)) + ...
    crossentropy(1-generatedPredictions, labels)/numel(generatedPredictions(:,:,1,1)));

% get d gradients
dGrads = dlgradient(dLoss, d.Learnables, "RetainData", true);
dLoss = extractdata(dLoss);

%% G update
% to save time I just use the existing result from d

% calculate g Losses
ganLoss = crossentropy(generatedPredictions, labels)/numel(generatedPredictions(:,:,1,1));
lossL1 = mean(abs(generated - target), 'all');
gLoss = options.Lambda*lossL1 + ganLoss;

% get g grads
gGrads = dlgradient(gLoss, g.Learnables);

% update g
g.Learnables = dOpt.update(g.Learnables, gGrads);
% update d
d.Learnables = gOpt.update(d.Learnables, dGrads);
% things for plotting
gLoss = extractdata(gLoss);
lossL1 = extractdata(lossL1);
ganLoss = extractdata(ganLoss);

images = {generated, input, target};
end

% The trained model is then can be utilized for the testing datset to convert it from low rsolution to high resolution.
low_resolution_Input = heldData; % the rest of the 15% dataset
low_resolution_Input = single(reshape(low_resolution_Input, sizeData));

Now feeding this low input to the trained model
generated_high_resoltion_Output = p2p.translate(p2pModel, low_resolution_Input);
generated_high_resoltion_Output = squeeze(extractdata(gather(generated_high_resoltion_Output)));


