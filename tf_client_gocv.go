// What it does:
//
// This example uses the Tensorflow (https://www.tensorflow.org/) deep learning framework
// to classify whatever is in front of the camera.
//
// Download the Tensorflow "Inception" model and descriptions file from:
// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
//
// Extract the tensorflow_inception_graph.pb model file from the .zip file.
//
// Also extract the imagenet_comp_graph_label_strings.txt file with the descriptions.
//
// How to run:
//
// 		go run ./cmd/tf-classifier/main.go 0 ~/Downloads/tensorflow_inception_graph.pb ~/Downloads/imagenet_comp_graph_label_strings.txt opencv cpu

// go run tf_client_gocv.go 0 .\modeltemp\saved_model.pb  .\modeltemp\labels.csv opencv cpu
// go run tf_client_gocv.go 0 .\modeltemp\tensorflow_inception_graph.pb  .\modeltemp\imagenet_comp_graph_label_strings.txt opencv cpu

//go run tf_client_gocv.go 0 .\TestModel\saved_model.pb  .\TestModel\labels.txt opencv cpu
//go run tf_client_gocv.go 0 .\teachablemodel\frozen_graph.pb  .\teachablemodel\labels.txt opencv cpu
//

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image"
	"image/color"
	"os"

	"gocv.io/x/gocv"
)

func main() {
	// if len(os.Args) < 4 {
	// 	fmt.Println("How to run:\ntf-classifier [camera ID] [modelfile] [descriptionsfile]")
	// 	return
	// }

	// parse args
	deviceID := 0
	//model := "./teachablemodel/frozen_graph.pb"
	model := "./model/frozen_model/frozen_graph.pb"
	descr := "./model/frozen_model/labels.txt"

	descriptions, err := readDescriptions(descr)
	if err != nil {
		fmt.Printf("Error reading descriptions file: %v\n", descr)
		return
	}

	// backend := gocv.NetBackendDefault
	// if len(os.Args) > 4 {
	// 	backend = gocv.ParseNetBackend(os.Args[4])
	// }

	// target := gocv.NetTargetCPU
	// if len(os.Args) > 5 {
	// 	target = gocv.ParseNetTarget(os.Args[5])
	// }

	// open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("Tensorflow Classifier")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// open DNN classifier
	//net := gocv.ReadNet(model, "")
	net := gocv.ReadNetFromTensorflow(model)
	if net.Empty() {
		fmt.Printf("Error reading network model : %v\n", model)
		return
	}
	defer net.Close()
	//net.SetPreferableBackend(gocv.NetBackendType(backend))
	//net.SetPreferableTarget(gocv.NetTargetType(target))

	status := "Ready"
	statusColor := color.RGBA{0, 255, 0, 0}
	fmt.Printf("Start reading device: %v\n", deviceID)

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		dstimg := gocv.NewMat()
		defer dstimg.Close()

		gocv.Flip(img, &dstimg, 1)

		dstimg2 := gocv.NewMat()
		defer dstimg2.Close()

		gocv.Resize(dstimg, &dstimg2, image.Pt(224, 224), 0, 0, 0)

		blob := gocv.BlobFromImage(dstimg2, 1.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), true, false)

		// feed the blob into the classifier
		net.SetInput(blob, "x")

		// run a forward pass thru the network
		//prob := net.Forward("Identity")
		prob := net.Forward("")

		i := 0
		var probstr bytes.Buffer
		for i < prob.Cols() {

			floatstr := fmt.Sprintf("prob (i=%d)[%f] ,,,,, ", i, prob.GetFloatAt(0, i))
			fmt.Println("prob.GetFloatAt(0, i)", prob.GetFloatAt(0, i))
			probstr.WriteString(floatstr)

			i = i + 1

		}

		fmt.Printf("------- prob.Data[%s]\n", probstr)

		fmt.Println("prob size", prob.Size())
		// reshape the results into a 1x1000 matrix
		probMat := prob.Reshape(1, 1)

		fmt.Println("probMat.Size", probMat.Size())
		//fmt.Println("probMat.ElemSize", probMat.ElemSize())

		fmt.Println("probMat.Data")

		i = 0

		var str bytes.Buffer
		for i < probMat.Cols() {

			floatstr := fmt.Sprintf("(i=%d)[%f] ,,,,, ", i, probMat.GetFloatAt(0, i))
			fmt.Println("probMat.GetFloatAt(0, i)", probMat.GetFloatAt(0, i))
			str.WriteString(floatstr)

			i = i + 1

		}

		fmt.Printf("------- probMat.Data[%s]\n", str)

		// determine the most probable classification
		minVal, maxVal, minLoc, maxLoc := gocv.MinMaxLoc(probMat)

		fmt.Println("----probMat", probMat)

		fmt.Printf("minVal[%f] , maxVal[%f] , minLoc[%v] , maxLoc[%v]\n", minVal, maxVal, minLoc, maxLoc)

		// display classification
		desc := "Unknown"
		if maxLoc.X < 1000 {
			desc = descriptions[maxLoc.X]
		}
		status = fmt.Sprintf("description: %v, maxVal: %v\n", desc, maxVal)
		fmt.Println("status", status)
		gocv.PutText(&dstimg2, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

		blob.Close()
		prob.Close()
		probMat.Close()

		break
		window.IMShow(dstimg2)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

// readDescriptions reads the descriptions from a file
// and returns a slice of its lines.
func readDescriptions(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}
