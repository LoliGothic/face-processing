import cv2

# 顔（正面）を検出できるカスケード識別器の学習済みファイルをGithubから持ってきた
cascade_path = "./haarcascade_frontalface_default.xml"
# カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)


def main():
    # カメラを選択．カメラが複数あるときは0を1や2にする
    cap = cv2.VideoCapture(0)
  
    # カメラが開かれている間処理を繰り返す
    while cap.isOpened():
        try:
            # RGBAで画像を読み込む
            face_img = cv2.imread('./image_alien-face.png', cv2.IMREAD_UNCHANGED)
            # フレームを配列として格納
            _, frame = cap.read()
            # 高速に顔検出するできるため，フレームをグレースケールに変換
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 顔認識の実行
            facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

            # 検出した場合
            if len(facerect) > 0:
                for rect in facerect:
                    # 首などに誤反応がでるため，サイズが100以下の誤反応は無視する
                    if rect[2] < 100 and rect[3] < 100:
                        rect.clear()
                        break

                    # 画像を顔に合わせて大きくする．大きくしすぎると重くなる
                    face_img = cv2.resize(face_img, ((int)(rect[2]*1.2), (int)(rect[3]*1.5)), cv2.IMREAD_UNCHANGED)
                    
                    # x座標をオフセット
                    rect[0] -= rect[2]*0.15
                    # y座標をオフセット
                    rect[1] -= rect[3]*0.3
                    
                    # 透過処理をして顔に画像を重ねる
                    # face_img[:, :, :3]はRGB．face_img[:, :, 3:]は透過度
                    frame[rect[1]:rect[1]+face_img.shape[0],
                          rect[0]:rect[0]+face_img.shape[1]] = frame[rect[1]:rect[1]+face_img.shape[0],
                                                                    rect[0]:rect[0]+face_img.shape[1]] * (1 - face_img[:, :, 3:] / 255) + face_img[:, :, :3] * (face_img[:, :, 3:] / 255)
                    cv2.imshow('cv2', frame)

        except:
            pass
        # 「q」キーが押されたら終了する
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # カメラを閉じる
    cap.release()

if __name__ == '__main__':
    main()