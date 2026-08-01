
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

t = 'path to your dataset.csv'
hose = pd.read_csv(t)
print(hose)
print(hose.info())

hose['time'] = pd.to_datetime(hose['time'])
hose['time'].dtype
hose = hose.set_index('time').sort_index()
#hose.reset_index(inplace=True)

"""
----------------------
2. Các bước tính toán
(a) Tính lợi suất
----------------------
"""
# tính lợi nhuận (return) hàng ngày và xóa những dữ liệu thiếu
returns = np.log(hose/hose.shift(1))
returns  = returns.dropna()
print(returns)
# Kiểm tra các giá trị lạ không dataframe
print("Có NaN không? ", returns.isna().any().any())
print("Có giá trị 0 không? ", (returns == 0).any().any())
print("Có giá trị inf hoặc -inf không? ", np.isinf(returns.values).any())
#Loại bỏ inf / -inf bằng cách thay thành NaN, rồi drop:
returns = returns.replace([np.inf, -np.inf], np.nan)
returns = returns.dropna()
# print lại kiểm tra
returns.to_excel('returns.xlsx')

"""
----------------------
2. Các bước tính toán
(b) Ước lượng beta rolling window (60 ngày)
-----------------------
"""
# -----------------------------
# 3. Rolling regression (CAPM beta)
# -----------------------------
window = 60  # rolling window = 60 ngày

# Tách VNINDEX (thị trường) và cổ phiếu
Rm = returns['VNINDEX']
Ri = returns.drop(columns=['VNINDEX'])

# Rolling covariance của từng cổ phiếu với thị trường
rolling_cov = Ri.rolling(window).cov(Rm)

# Rolling variance của thị trường
rolling_var = Rm.rolling(window).var()

# Tính rolling beta: mỗi cột là beta theo thời gian
rolling_beta = rolling_cov.div(rolling_var, axis=0)
betas = rolling_beta.dropna()
# Hiển thị beta động của 5 cổ phiếu đầu tiên
print(rolling_beta)
print(betas)


"""
----------------------
2. Các bước tính toán
(c) Độ lệch chuẩn chéo
(d) Biến quan sát y_t = log (độ lệch chuẩn chéo)
----------------------
"""
# -----------------------------
# 4. Cross-sectional std + log
# -----------------------------
std_betas = betas.std(axis=1, skipna=True)
y_t = np.log(std_betas)
y_t = y_t.dropna()
print(y_t) 
# y_t là log của độ lệch chuẩn 
# y_t càng cao ⇒ độ lệch chuẩn của beta cao ⇒ các beta của cổ phiếu khác biệt nhiều, ít bầy đàn.
# y_t càng thấp (âm sâu) ⇒ beta các cổ phiếu giống nhau ⇒ có dấu hiệu bầy đàn.


# Tính thủ công std (chọn 1 ngày để kiểm tra giữa manual và pandas std)
x = betas.loc["2012-08-22"].values
n = len(x)

mean_x = x.mean()
var_sample = ((x - mean_x)**2).sum() / (n - 1)
std_sample = np.sqrt(var_sample)

print("Manual std:", std_sample)
print("Pandas std:", betas.loc["2012-08-22"].std(ddof=1))

"""
----------------------
2. Các bước tính toán
(e) Mô hình không gian trạng thái
----------------------
"""
# -----------------------------
# 5. Mô hình không gian trạng thái
# -----------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel
class HerdingStateSpace(MLEModel):
    def __init__(self, endog):
        super().__init__(endog, k_states=1, initialization="approximate_diffuse")
        """
        HerdingStateSpace kế thừa từ MLEModel của statsmodels, nghĩa là đây là một mô hình state-space 1 chiều (1 state).
        endog là chuỗi dữ liệu quan sát y_t (ví dụ: biến đại diện cho herding).
        k_states=1 → có 1 biến trạng thái tiềm ẩn H_t
        initialization="approximate_diffuse" → khởi tạo trạng thái ban đầu bằng diffuse prior (dùng khi trạng thái ban đầu không biết chắc).
        """
        # Observation eq: y_t = mu + H_t + v_t
        self['design'] = np.array([[1.0]])      # matrix Z: y_t depends on H_t linearly
        self['obs_intercept'] = np.array([0.0]) # mu = 0 ban đầu
        self['obs_cov'] = np.array([[1.0]])     # Var(v_t) = 1 ban đầu
        
        # State eq: H_t = phi * H_{t-1} + eta_t
        self['transition'] = np.array([[0.5]])  # phi
        self['selection'] = np.array([[1.0]])   # state selection
        self['state_cov'] = np.array([[1.0]])   # Var(eta_t) = 1

    def update(self, params, transformed=True, **kwargs):
        params = super().update(params, transformed, **kwargs)
        # update dùng để gán tham số thực tế từ tối ưu likelihood vào ma trận state-space
        # tham số gồm:
        mu, phi, sigma_v, sigma_eta = params
        self['obs_intercept', 0] = mu           # intercept của observation eq.
        self['transition', 0, 0] = phi          # AR(1) coefficient của state eq
        self['obs_cov', 0, 0] = sigma_v**2      # độ lệch chuẩn của quan sát, lưu vào obs_cov.
        self['state_cov', 0, 0] = sigma_eta**2  # độ lệch chuẩn của state noise, lưu vào state_cov

    @property
    def start_params(self):
        return np.array([0.0, 0.9, 0.1, 0.1])  # mu, phi, sigma_v, sigma_eta
        # Đây là giá trị ban đầu để optimizer bắt đầu tìm maximum likelihood.

# Fit model
model = HerdingStateSpace(y_t.values)
res = model.fit() # ước lượng các tham số (mu, phi, sigma_v, sigma_eta) bằng maximum likelihood.

print(res.summary())

# Đọc kết quả:
## param.0 = mu (hằng số phương trình quan sát) -> KHONG CO Y NGHIA THONG 
## param.1 = phi (hệ số tự hồi quy trong trạng thái) => độ bền của bầy đàn 
    ## (gần 1, càng bền, quán tính mạnh; gần 0 bầy đàn biến động nhanh, không ổn định )
## param.2 = sigma v -> KHONG CO Y NGHIA THONG 
## param.3 = sigma eta => bầy đàn biến động theo thời gian
# KHONG CO Y NGHIA THONG KE XET THEO P>|Z|

# -----------------------------
# Vẽ so sánh y_t (observed) vs fitted values
# -----------------------------
fitted_values = res.fittedvalues 
#DA ƯỚC LƯỢNG RA HÊT HỆ SỐ -> BỎ PHẦN SAI 

plt.figure(figsize=(8,4))
plt.plot(y_t.index, y_t, label="Observed y_t (log Std Beta)", color="black")
plt.plot(y_t.index, fitted_values, label="Fitted values (from state-space)", color="red")
plt.title("Observed vs Fitted: State-Space Herding Model")
plt.xlabel("Time")
plt.ylabel("log(Std Beta)")
plt.legend()
plt.show()

# -----------------------------
# 6. Kết quả: h_t = 1 - exp(H_t)
# -----------------------------
smoothed_H = res.smoothed_state[0]  # ước lượng H_t dựa trên toàn bộ dữ liệu (smoothed estimate).
h_t = 1 - np.exp(smoothed_H) # h_t
h_t_series = pd.Series(h_t, index=y_t.index)
print(h_t)
print(h_t_series.head())

# đọc kết quả: h_t tiến gần 1 => càng chứng tỏ bằng chứng mạnh mẽ về hành vi bầy đàn

# Vẽ kết quả
plt.figure(figsize=(8,4))
plt.plot(h_t_series, label="Herding intensity h_t")
plt.title("Herding Behavior in Vietnam Stock Market")
plt.xlabel("Time")
plt.ylabel("Herding Intensity")
plt.legend()
plt.show()
