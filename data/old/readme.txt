JAFFE:
    - JAFFE dataset

dLBP45_JAFFE:
    -bilateralFilter(image, 5, 30, 20)
    -dLBP(45)
    -LBPH(18, 18)
    -2 images per emotion per person for training (total=140), the rest for test(73)

kirsch_hvnLBP_JAFFE:
    -bilateralFilter(image, 5, 30, 20)
    -resize(77, 77)
    -kirsch
    -hvnLBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

kirsch_hvnLBP_JAFFE1:
    -5x bilateralFilter(image, 5, 30, 20)
    -resize(77, 77)
    -kirsch
    -hvnLBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

kirsch_hvnLBP_JAFFE2:
    -5x bilateralFilter(image, 9, 250, 250)
    -resize(77, 77)
    -kirsch
    -hvnLBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

kirsch_LBP_JAFFE2:
    -5x bilateralFilter(image, 9, 250, 250)
    -resize(75, 75)
    -kirsch
    -LBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

hvnLBP_JAFFE:
    -bilateralFilter(image, 5, 30, 20)
    -resize(77, 77)
    -hvnLBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

hvnLBP_JAFFE1:
    -5x bilateralFilter(image, 5, 30, 20)
    -resize(77, 77)
    -hvnLBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

hvnLBP_JAFFE2:
    -5x bilateralFilter(image, 9, 250, 250)
    -resize(77, 77)
    -hvnLBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

LBP_JAFFE:
    -bilateralFilter(image, 5, 30, 20)
    -resize(75, 75)
    -LBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

LBP_JAFFE1:
    -5x bilateralFilter(image, 5, 30, 20)
    -resize(75, 75)
    -LBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)

LBP_JAFFE2:
    -5x bilateralFilter(image, 9, 250, 250)
    -resize(75, 75)
    -LBP
    -LBPH(25, 25)
    -train_test_split(data, labels, test_size=0.33, random_state=42)