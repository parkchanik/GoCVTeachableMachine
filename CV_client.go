package main

import (
	"fmt"
	"os"

	"gocv.io/x/gocv"
)

func main() {

	filename := os.Args[1]
	window := gocv.NewWindow("Hello")
	img := gocv.IMRead(filename, gocv.IMReadColor)
	if img.Empty() {
		fmt.Printf("Error reading image from: %v\n", filename)
		return
	}
	for {
		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}

}
