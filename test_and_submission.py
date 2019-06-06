test_path = '../input/quickdraw-doodle-recognition/test_simplified.csv'
testset = pd.read_csv(test_path)

def test_generator(batch_size=680):
    start = 0
    end = 0
    while True:
        end = start + BATCH_SIZE

        batch_img = []
        for i in range(start, end):
            img = drawing_to_img(testset.drawing[i], LINE_WIDTH)
            img = img_preprocess(img)

            batch_img.append(img)
        
        start += end
        yield np.array(batch_img)


n_test = sum(1 for l in open(val_test)) - 1
test_gen = test_generator(test_path, BATCH_SIZE)

output = model.predict_generator(test_gen, steps=ceil(n_test/BATCH_SIZE), verbose=1)

result = pd.DataFrame()
for i in range(n_test):

    output_dic = {}
    for i, o in enumerate(output[i]):
        output_dic[o] = i

    pred = []
    for j, o in enumerate(sorted(output[i], reverse=True)):
        pred.append(class_names[output_dic[o]])
        if j == 2:
            break
    
#     print(testset[i], end='')
#     for j in range(3):
#         print(pred[j], end='')
#     print()
#     break
    
    result.append(testset[i], pred)

result.to_csv('submission.csv')
