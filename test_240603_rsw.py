"""
test_240603_rsw.py : 가장 최신 구동 파일
logs.txt : samples/GenICamParameters.py 이용, SDK 이용해 얻을 수 있는 모든 파라미터 목록
- 형식 : Category:Parameter_name, Type: Value

* 240603 수정사항*
1. waterfall.py 참고해 waterfall 나오도록 변경
    - extract_band_image_3 기반으로 재작성
2. SDK 관련 모든 함수에 type hint 적용
3. output 파일들이 outputs 폴더에 저장되도록 함 (hdr, npy)
4. envi header에서 식별이 불가능한 몇몇 변수를 제외하고 sample.hdr와 동일한 형태가 되도록 작성, 알 수 없는 값은 "?"로 초기화
5. wavelegnth, fwhm 상수들은 constants.py 파일에 모아두었음.
6. 처음에 0 눌러서 (device selection) 시작하는 작업 없앰. # 만일 되돌릴려면 lib/PvSampleUtils.py의 96 line uncomment, 97 line comment하면 됌.
7. .raw 데이터로 저장 -> .npy로 변경
"""

import eBUS as eb
import lib.PvSampleUtils as psu
import numpy as np
import cv2
from datetime import datetime, timezone
from typing import List
import constants

BUFFER_COUNT = 16
kb = psu.PvKb()

BANDS = constants.BANDS

def connect_to_device(connection_ID) -> eb.PvDeviceGEV:
    print("Connecting to device.")
    result, device = eb.PvDevice.CreateAndConnect(connection_ID)
    if device is None:
        print(
            f"Unable to connect to device: {result.GetCodeString()} ({result.GetDescription()})"
        )
    return device

def get_info(device: eb.PvDeviceGEV, target: str):
    parameters = device.GetParameters()
    gen_parameter = parameters.Get(target)
    _, category = gen_parameter.GetCategory()
    _, gen_parameter_name = gen_parameter.GetName()
    _, gen_type = gen_parameter.GetType()
    if eb.PvGenTypeInteger == gen_type:
        result, value = gen_parameter.GetValue()
    elif eb.PvGenTypeEnum == gen_type:
        result, value = gen_parameter.GetValueString()
    elif eb.PvGenTypeBoolean == gen_type:
        result, value = gen_parameter.GetValue()
    elif eb.PvGenTypeString == gen_type:
        result, value = gen_parameter.GetValue()
    elif eb.PvGenTypeFloat == gen_type:
        result, value = gen_parameter.GetValue()
    return value

def open_stream(connection_ID) -> eb.PvStreamGEV:
    print("Opening stream from device.")
    result, stream = eb.PvStream.CreateAndOpen(connection_ID)
    if stream is None:
        print(
            f"Unable to stream from device. {result.GetCodeString()} ({result.GetDescription()})"
        )
    return stream


def configure_stream(device, stream) -> None:
    if isinstance(device, eb.PvDeviceGEV):
        device.NegotiatePacketSize()
        device.SetStreamDestination(stream.GetLocalIPAddress(), stream.GetLocalPort())


def configure_stream_buffers(device, stream) -> List[eb.PvBuffer]:
    buffer_list = []
    size = device.GetPayloadSize()
    buffer_count = min(stream.GetQueuedBufferMaximum(), BUFFER_COUNT)

    for _ in range(buffer_count):
        pvbuffer = eb.PvBuffer()
        pvbuffer.Alloc(size)
        buffer_list.append(pvbuffer)

    for pvbuffer in buffer_list:
        stream.QueueBuffer(pvbuffer)

    print(f"Created {buffer_count} buffers")
    return buffer_list


## mono12packed라는 픽셀 타입을 사용
def convert_mono12_packed(
    image_data: np.ndarray, width: int, height: int
) -> np.ndarray:
    """
    mono12packed 형식의 이미지 구조를 12bit pixel data -> 16bit 형식으로 변환
    :param image_data:
    :param width:
    :param height:
    :return:
    """

    raw_data = np.frombuffer(image_data, np.uint8)  ## 1차원에서 2차원 배열로 변환
    raw_data = raw_data.reshape(
        (height, (width * 3) // 2)
    )  ## 기존 이미지를 3/2로 스케일링
    # Width: 1024, Height: 224, Raw Data Shape: (224, 1536)

    expanded_16bit_data = np.zeros(shape=(height, width), dtype=np.uint16)

    # 3바이트(RGB)를 읽고, 2개 16bit 픽셀로 변환
    byte_0 = raw_data[:, 0::3].astype(np.uint16)
    byte_1 = raw_data[:, 1::3].astype(np.uint16)
    byte_2 = raw_data[:, 2::3].astype(np.uint16)

    # 12bit로 변경
    expanded_16bit_data[:, 0::2] = (byte_0 << 4) | (byte_2 & 0xF)
    expanded_16bit_data[:, 1::2] = (byte_1 << 4) | (byte_2 >> 4)
    print(
        f"convert_mono12_packed({expanded_16bit_data.ndim}, {expanded_16bit_data.shape}, {expanded_16bit_data.dtype})"
    )
    return expanded_16bit_data


def pixel_format_to_cv2_format(pixel_format) -> str:
    format_dict = {
        eb.PvPixelMono8: "MONO8",
        eb.PvPixelRGB8: "RGB8",
        eb.PvPixelMono12Packed: "MONO12PACKED",  # Pixel Type for SPECIM
        # PvPixelMono16: 'MONO16',
        # PvPixelRGB16: 'RGB16', # Define actual type as needed
    }
    return format_dict.get(pixel_format, "Unsupported")


def waterfall_update(waterfall_image, new_row):
    # Check if new_row has the same width as the waterfall_image
    if new_row.shape[1] != waterfall_image.shape[1]:
        # Reshape new_row to have the same width as waterfall_image
        new_row = new_row.reshape(1, waterfall_image.shape[1], 3)

    # 원본 ver.
    # waterfall_image = np.delete(waterfall_image, 0, axis=0)
    # waterfall_image = np.vstack((waterfall_image, new_row))

    # 수정 ver.
    waterfall_image = np.roll(waterfall_image, -1, axis=0)
    waterfall_image[-1, :, :] = new_row

    return waterfall_image


def extract_band_image(data, width, height, band_index, pixel_length):

    ## 원본 ver.
    line_length = width * pixel_length
    band_data = data.reshape(-1)
    start_index = line_length * (band_index - 1)
    band_image = band_data[start_index : start_index + width].reshape(1, -1)

    ## 수정 ver.
    # band_data = data[band_index, :]
    # band_image = band_data.reshape(1, -1)

    return band_image


def extract_band_image_2(data, bands):
    """
    밴드 인덱스를 이용해 밴드 이미지 추출하는 기능 함수
    :param data: 디바이스를 통해 수신받은 이미지 객체
    :param bands: 밴드 인덱스가 담긴 리스트
    :return:

    """
    # b_image, g_image, r_image = (
    #     data[bands[0], :].reshape(1, -1),
    #     data[bands[1], :].reshape(1, -1),
    #     data[bands[2], :].reshape(1, -1)
    # )

    b_image, g_image, r_image = (
        data[:, bands[0]].reshape(-1, 1),
        data[:, bands[1]].reshape(-1, 1),
        data[:, bands[2]].reshape(-1, 1),
    )

    # return [data[:, band] for band in bands]
    return b_image, g_image, r_image


def extract_band_image_3(
    data: np.ndarray, width: int, height: int, band_index: int, pixel_length: int
):
    """

    :param data: 이미지 데이터
    :param width: 너비
    :param height: 높이
    :param start_pixel: 밴드 시작 위치
    :param pixel_length: 픽셀 사이즈
    :return:
    """

    line_length = width * pixel_length  # 너비
    start_index = (band_index - 1) * line_length  # 밴드 시작 위치
    end_index = start_index + (line_length * pixel_length)  # 밴드 시작 위치 +

    # print(f"start_index: {start_index}, end_index: {end_index}")

    if end_index > data.size:
        end_index = data.size

    band_data = data[start_index:end_index]
    if band_data.size == 0:
        print(f"Error: band_data is empty for start_index {start_index}")

    # Reshape and transpose band data
    # print(f"Size: {band_data.size}, {band_data.shape}")
    band_data = np.reshape(band_data, (-1, width))
    # print(f"전치 후 {band_data.shape}")

    return band_data


def acquire_images(device, stream, output_filename_prefix="hyperspectral_data"):
    """
    :param device: 디바이스 객체
    :param stream: 디바이스 내 스트림 객체
    :param output_filename_prefix:
    :return:
    """
    t1 = datetime.now(timezone.utc)
    # 디바이스 파라미터 정보를 가져온다.
    device_params: eb.PvGenParameterArray = device.GetParameters()

    start: eb.PvGenCommand = device_params.Get("AcquisitionStart")
    stop: eb.PvGenCommand = device_params.Get("AcquisitionStop")
    stream_params: eb.PvGenParameterArray = stream.GetParameters()
    frame_rate: eb.PvGenFloat = stream_params.Get("AcquisitionRate")
    bandwidth: eb.PvGenFloat = stream_params["Bandwidth"]
    exposure_time: eb.PvGenFloat = device_params.Get("ExposureTime")
    # print each parameter
    print(f"frame_rate: {frame_rate.GetValue()}")
    print(f"bandwidth: {bandwidth.GetValue()}")
    print(f"exposure_time: {exposure_time.GetValue()}")

    # 장치 파라미터가 저장된 객체 및 필요한 파라미터를 가져오기
    shutterOpen: eb.PvGenInteger = device_params.Get("MotorShutter_PulseRev")
    shutterClose: eb.PvGenInteger = device_params.Get("MotorShutter_PulseFwd")

    binning_h: eb.PvGenInteger = device_params.Get("BinningHorizontal")  # 1
    binning_v: eb.PvGenInteger = device_params.Get("BinningVertical")  # 2

    # 현재 shutterOpen 값을 가져와 1 증가시키기
    result, shutterOpen_value = shutterOpen.GetValue()
    # 오류가 없는지 확인
    if not result.IsOK():
        print(f"Failed to get current value: {result}")
        exit(1)

    print("Current shutterOpen value:", shutterOpen_value)

    shutterOpen_value += 1

    # 증가된 값을 설정
    result = shutterOpen.SetValue(shutterOpen_value)
    result, shutterOpen_value = shutterOpen.GetValue()

    # 센서값이 210이 넘어가면 다시 센서 값을 200으로 맞춰준다.
    if shutterOpen_value >= 210:
        shutterOpen_value = 200
        result = shutterOpen.SetValue(shutterOpen_value)

    # 오류가 없는지 확인
    if not result.IsOK():
        print(f"Failed to set new value: {result}")
        exit(1)

    print(f"Updated shutterOpen value to: {shutterOpen_value}")

    # 현재 shutterClose 값을 가져와 1 감소시키기
    result, shutterClose_value = shutterClose.GetValue()

    # 오류가 없는지 확인
    if not result.IsOK():
        print(f"Failed to get current value: {result}")
        exit(1)

    print("Enabling streaming and sending AcquisitionStart command.")
    device.StreamEnable()
    start.Execute()

    width = 1024  # Assume width of the image
    height = 500  # Waterfall image height
    pixel_length = 1  # 1 if ushort array else 2

    waterfall_image = np.zeros(
        (height, width, 3), dtype=np.uint8
    )  # Initialize waterfall image
    collected_images = []  # List to store collected images

    print("\n<press a key to stop streaming>")
    kb.start()
    lines = 0
    while not kb.is_stopping():
        # print(f"Updated shutterClose value to: {shutterClose_value}")
        # print(f"Updated shutterOpen value to: {shutterOpen_value}")
        result, pvbuffer, operational_result = stream.RetrieveBuffer(1000)
        if result.IsOK():
            if operational_result.IsOK():
                print(f"BlockID: {pvbuffer.GetBlockID()}")

                pixel_type = pvbuffer.GetImage().GetPixelType()
                cv_pixel_type = pixel_format_to_cv2_format(pixel_type)

                if cv_pixel_type == "MONO12PACKED":
                    image = pvbuffer.GetImage()
                    width = image.GetWidth()
                    height = image.GetHeight()
                    data = convert_mono12_packed(image.GetDataPointer(), width, height)
                    lines += 1
                    b_image = extract_band_image_3(
                        data.flatten(), width, height, BANDS.b, pixel_length
                    )
                    g_image = extract_band_image_3(
                        data.flatten(), width, height, BANDS.g, pixel_length
                    )
                    r_image = extract_band_image_3(
                        data.flatten(), width, height, BANDS.r, pixel_length
                    )

                    # print(f"b_image shape: {b_image.shape}")
                    # print(f"g_image shape: {g_image.shape}")
                    # print(f"r_image shape: {r_image.shape}")
                    # print(f"data shape: {data.shape}")

                    # Check extracted images
                    if b_image.size == 0 or g_image.size == 0 or r_image.size == 0:
                        print("Error: Empty image data extracted.")
                        continue

                    # Merge BGR
                    ## -------------예시-------------------------
                    merged_image = cv2.merge([b_image, g_image, r_image])
                    ## -------------예시-------------------------
                    merged_image = np.uint8(merged_image / 16)  # Convert to 8-bit image
                    collected_images.append(merged_image)  # Collect images

                    waterfall_image = waterfall_update(waterfall_image, merged_image)
                    cv2.imshow("Waterfall", waterfall_image)

                    if cv2.waitKey(1) & 0xFF == ord("q"):

                        # 값을 1 감소
                        shutterClose_value += 1

                        # 감소된 값을 설정
                        result = shutterClose.SetValue(shutterClose_value)

                        # 오류가 없는지 확인
                        if not result.IsOK():
                            print(f"Failed to set new value: {result}")
                            exit(1)

                        print(f"Updated shutterClose value to: {shutterClose_value}")
                        # save waterfall image with numpy array
                        # np.save("waterfall_image.npy", waterfall_image)
                        break

                    else:
                        # print(f"Unsupported pixel format: {pixel_type}")
                        stream.QueueBuffer(pvbuffer)
                        continue

                    stream.QueueBuffer(pvbuffer)

                else:
                    print(
                        f"Operational result not OK: {operational_result.GetCodeString()}",
                        end="\r",
                    )
            else:
                print(f"RetrieveBuffer failed: {result.GetCodeString()}", end="\r")

            if kb.kbhit():
                kb.getch()
                break

            ## 수정 ver. (added)
            # except Exception as e:
            #     print(e)

            # 값을 1 감소
            shutterClose_value += 1

            # # 감소된 값을 설정
            result = shutterClose.SetValue(shutterClose_value)
            if shutterClose_value >= 210:
                shutterClose_value = 200
                result = shutterClose.SetValue(shutterClose_value)
            break

    shutterClose_value += 1

    # 감소된 값을 설정
    result = shutterClose.SetValue(shutterClose_value)
    if shutterClose_value >= 210:
        shutterClose_value = 200
        result = shutterClose.SetValue(shutterClose_value)

    # 오류가 없는지 확인
    if not result.IsOK():
        print(f"Failed to set new value: {result}")
        exit(1)

    

    kb.stop()
    cv2.destroyAllWindows()
    print("\nSending AcquisitionStop command to the device")
    stop.Execute()
    print("Disabling streaming on the device.")
    device.StreamDisable()
    print("Aborting buffers still in stream")
    stream.AbortQueuedBuffers()
    while stream.GetQueuedBufferCount() > 0:
        stream.RetrieveBuffer()

    # Convert collected images to a numpy array
    images = np.array(collected_images)

    if images.size == 0:
        print("No images were collected.")
        return
    t2 = datetime.now(timezone.utc)
    print("Shape of data:", data.shape)
    print("Shape of collected images:", images.shape)
    print("Shape of waterfall image:", waterfall_image.shape)
    # Save images to ENVI format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    header_filename = f"outputs/{output_filename_prefix}_{timestamp}.hdr"
    data_filename = f"outputs/{output_filename_prefix}_{timestamp}.npy"
    save_envi(data, device, header_filename, data_filename, times=[t1, t2], lines=lines, images=images)


def save_envi(data, device, header_filename, data_filename, **kwargs):
    raw_data = kwargs.get("raw_data")
    parameters = device.GetParameters()
    header = {
        "description": "Hyperspectral Image",
        "sensor type": get_info(device, "DeviceModelName"),
        "acquisition date": f'DATE(yyyy-mm-dd): {datetime.now().strftime("%Y-%m-%d")}',
        "Start Time": f'UTC TIME: {kwargs.get("times")[0].strftime("%H:%M:%S")}',
        "Stop Time": f'UTC TIME: {kwargs.get("times")[1].strftime("%H:%M:%S")}',
        "samples": data.shape[1],   # 1024
        "bands": data.shape[0], # 224
        "lines": kwargs.get("lines"),
        "errors": "?",
        "header offset": 0,
        "file type": "ENVI Standard",
        "data type": 12,  # 12 is the data type for unsigned 16-bit integer
        "interleave": "bil",    # ?
        "byte order": 0,        # ?
        "x start": "?",
        "y start": "?",
        "default bands": f"{{{BANDS.r}, {BANDS.g}, {BANDS.b}}}",
        "himg": f"{{1, {data.shape[1]}}}",
        "vimg": f"{{1, {data.shape[0]}}}",
        "hroi": f"{{1, {data.shape[1]}}}",
        "vroi": f"{{1, {data.shape[0]}}}",
        "fps": get_info(device, "AcquisitionFrameRate"),
        "fps_qpf": get_info(device, "AcquisitionFrameRate"), # What is qpf?
        "tint": "?",
        "binning": f"{{{get_info(device, 'BinningVertical')}, {get_info(device, 'BinningHorizontal')}}}",   # Check if order is right
        "trigger mode": get_info(device, "TriggerMode"),
        "fodis": "?",
        "sensorid": get_info(device, "DeviceSerialNumber"),
        "acquisitionwindow left": "?",
        "acquisitionwindow top": "?",
        "calibration pack": "?",
        "VNIR temperature": get_info(device, 'DeviceTemperature'),  # Same with temperature in sample.hdr, but need to check
        "temperature": f"{{\n{get_info(device, 'DeviceTemperature')}\n}}",
        "wavelength units": "Nanometers",
        # wave length 정보
        #  "wavelength": {397.66, 400.28, 402.90, 405.52, 408.13, 410.75, 413.37, 416.00, 418.62, 421.24, 423.86, 426.49,
        #                429.12, 431.74, 434.37, 437.00, 439.63, 442.26, 444.89, 447.52, 450.16, 452.79, 455.43, 458.06,
        #                460.70, 463.34, 465.98, 468.62, 471.26, 473.90, 476.54, 479.18, 481.83, 484.47, 487.12, 489.77,
        #                492.42, 495.07, 497.72, 500.37, 503.02, 505.67, 508.32, 510.98, 513.63, 516.29, 518.95, 521.61,
        #                524.27, 526.93, 529.59, 532.25, 534.91, 537.57, 540.24, 542.91, 545.57, 548.24, 550.91, 553.58,
        #                556.25, 558.92, 561.59, 564.26, 566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583.00, 585.68,
        #                588.36, 591.04, 593.73, 596.41, 599.10, 601.78, 604.47, 607.16, 609.85, 612.53, 615.23, 617.92,
        #                620.61, 623.30, 626.00, 628.69, 631.39, 634.08, 636.78, 639.48, 642.18, 644.88, 647.58, 650.29,
        #                652.99, 655.69, 658.40, 661.10, 663.81, 666.52, 669.23, 671.94, 674.65, 677.36, 680.07, 682.79,
        #                685.50, 688.22, 690.93, 693.65, 696.37, 699.09, 701.81, 704.53, 707.25, 709.97, 712.70, 715.42,
        #                718.15, 720.87, 723.60, 726.33, 729.06, 731.79, 734.52, 737.25, 739.98, 742.72, 745.45, 748.19,
        #                750.93, 753.66, 756.40, 759.14, 761.88, 764.62, 767.36, 770.11, 772.85, 775.60, 778.34, 781.09,
        #                783.84, 786.58, 789.33, 792.08, 794.84, 797.59, 800.34, 803.10, 805.85, 808.61, 811.36, 814.12,
        #                816.88, 819.64, 822.40, 825.16, 827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52, 847.29,
        #                850.06, 852.83, 855.60, 858.37, 861.14, 863.92, 866.69, 869.47, 872.25, 875.03, 877.80, 880.58,
        #                883.37, 886.15, 888.93, 891.71, 894.50, 897.28, 900.07, 902.86, 905.64, 908.43, 911.22, 914.02,
        #                916.81, 919.60, 922.39, 925.19, 927.98, 930.78, 933.58, 936.38, 939.18, 941.98, 944.78, 947.58,
        #                950.38, 953.19, 955.99, 958.80, 961.60, 964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27,
        #                984.09, 986.90, 989.72, 992.54, 995.35, 998.17, 1000.99, 1003.81},
        # "fwhm": {2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.62, 2.63, 2.63, 2.63, 2.63, 2.63, 2.63,
        #          2.63, 2.63, 2.63, 2.63, 2.63, 2.64, 2.64, 2.64, 2.64, 2.64, 2.64, 2.64, 2.64, 2.64, 2.64, 2.64, 2.65,
        #          2.65, 2.65, 2.65, 2.65, 2.65, 2.65, 2.65, 2.65, 2.65, 2.65, 2.66, 2.66, 2.66, 2.66, 2.66, 2.66, 2.66,
        #          2.66, 2.66, 2.66, 2.66, 2.67, 2.67, 2.67, 2.67, 2.67, 2.67, 2.67, 2.67, 2.67, 2.67, 2.67, 2.68, 2.68,
        #          2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.69, 2.69, 2.69, 2.69, 2.69, 2.69, 2.69, 2.69,
        #          2.69, 2.69, 2.70, 2.70, 2.70, 2.70, 2.70, 2.70, 2.70, 2.70, 2.70, 2.70, 2.70, 2.71, 2.71, 2.71, 2.71,
        #          2.71, 2.71, 2.71, 2.71, 2.71, 2.71, 2.71, 2.72, 2.72, 2.72, 2.72, 2.72, 2.72, 2.72, 2.72, 2.72, 2.72,
        #          2.72, 2.73, 2.73, 2.73, 2.73, 2.73, 2.73, 2.73, 2.73, 2.73, 2.73, 2.73, 2.74, 2.74, 2.74, 2.74, 2.74,
        #          2.74, 2.74, 2.74, 2.74, 2.74, 2.74, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.75, 2.76,
        #          2.76, 2.76, 2.76, 2.76, 2.76, 2.76, 2.76, 2.76, 2.76, 2.76, 2.77, 2.77, 2.77, 2.77, 2.77, 2.77, 2.77,
        #          2.77, 2.77, 2.77, 2.77, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.78, 2.79, 2.79,
        #          2.79, 2.79, 2.79, 2.79, 2.79, 2.79, 2.79, 2.79, 2.79, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80, 2.80,
        #          2.80, 2.80, 2.80, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.81, 2.82, 2.82, 2.82,
        #          2.82, 2.82, 2.82, },
        "wavelength": constants.WAVELENGTH,
        "fwhm": constants.FWHM,
    }

    # Save header (.hdr) file
    with open(header_filename, "w") as hdr_file:
        hdr_file.write("ENVI\n")
        for key, value in header.items():
            hdr_file.write(f"{key} = {value}\n")

    # Save data (.raw) file
    np.save(data_filename, kwargs.get("images"))


print("PvStreamSample:")

# 장비 연결 IP를 리턴해준다.
connection_ID: str = psu.PvSelectDevice()

if connection_ID is not None:
    device: eb.PvDeviceGEV = connect_to_device(connection_ID)

    if device:
        stream: eb.PvStream = open_stream(connection_ID)
        if stream:
            configure_stream(device, stream)
            buffer_list = configure_stream_buffers(device, stream)
            acquire_images(device, stream)
            buffer_list.clear()
            print("Closing stream")
            stream.Close()
            eb.PvStream.Free(stream)

        print("Disconnecting device")
        device.Disconnect()
        eb.PvDevice.Free(device)

print("<press a key to exit>")
kb.start()
kb.getch()
kb.stop()
