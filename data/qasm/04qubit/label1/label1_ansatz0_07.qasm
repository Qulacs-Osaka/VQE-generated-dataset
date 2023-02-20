OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.045482341762445824) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.14382510735788925) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.028586385807914106) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.15294916908424783) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.16324358222808374) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13470479678079006) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.022656318884683387) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.13189119445790232) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.24212623770889824) q[3];
cx q[2],q[3];
rz(-0.15154885034091473) q[0];
rz(0.07530883510090301) q[1];
rz(0.4310189578475142) q[2];
rz(0.23508491609790613) q[3];
rx(-0.632283465970715) q[0];
rx(-1.097678838478166) q[1];
rx(-0.5081530329671289) q[2];
rx(-0.7228150700661315) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12966795458325955) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.20569707003331697) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.08425149950642676) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4829937031754602) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.4638828724067563) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.051736348795302575) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2059769137172661) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3348342565565071) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04759813499351205) q[3];
cx q[2],q[3];
rz(-0.019250871833736798) q[0];
rz(-0.04087637619439449) q[1];
rz(0.1166923941190898) q[2];
rz(-0.01022533453682837) q[3];
rx(-0.8350949717276261) q[0];
rx(-1.0513261522569655) q[1];
rx(-0.427491028522313) q[2];
rx(-0.513679134845723) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1563241894455624) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0370112008779499) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08766678065805643) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.5053015329847905) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.5265183634929068) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.06660740133916584) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.3587533187066533) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.6771006220768457) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11390780702180009) q[3];
cx q[2],q[3];
rz(0.18629145256699378) q[0];
rz(-0.2237784035913378) q[1];
rz(-0.09391204238370782) q[2];
rz(-0.07155659423063124) q[3];
rx(-0.8742928760946732) q[0];
rx(-0.9017369929544511) q[1];
rx(-0.3872510841547048) q[2];
rx(-0.4892171193516897) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07749471432890004) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.13830594367831833) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.22465979513150758) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7279307835363868) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.6503614922580837) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.10926737395056522) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.45116518301584024) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.6440952383095341) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0971970877844131) q[3];
cx q[2],q[3];
rz(0.16592224065811162) q[0];
rz(0.1310263851615559) q[1];
rz(0.035572482090121156) q[2];
rz(-0.01175889673890837) q[3];
rx(-0.9275610118535931) q[0];
rx(-0.8130114920391577) q[1];
rx(-0.5335193985119455) q[2];
rx(-0.568199252359239) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.16656851286342053) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.19534670520606412) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.05305681246425181) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7651317852288366) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.42302911932380854) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0570675779105357) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5249281910401257) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.5040181910040208) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.15981275067696707) q[3];
cx q[2],q[3];
rz(0.13751693923767025) q[0];
rz(0.0709025581228267) q[1];
rz(0.20638314435600452) q[2];
rz(0.27075156620517693) q[3];
rx(-0.9406755679549119) q[0];
rx(-0.6858940881359037) q[1];
rx(-0.6619930414313571) q[2];
rx(-0.5379219446222974) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.22990102902005852) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.30283916686962603) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.22393410804211036) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.7124275897234491) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.46759619583541556) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.13628190516927252) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.43988537636262326) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.4469009296811259) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.46018990068186655) q[3];
cx q[2],q[3];
rz(0.10869636978677424) q[0];
rz(0.22083903612194328) q[1];
rz(0.23638853347343183) q[2];
rz(0.4505094007286769) q[3];
rx(-0.9650415955518933) q[0];
rx(-0.555471357617376) q[1];
rx(-0.6876631438968444) q[2];
rx(-0.5298526553382763) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1931723546630574) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.15608805483657287) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.022261519285010887) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.4854701146784529) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.3790757856978726) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3107156686937296) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.42275649890498124) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.3647247128573858) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.36718827056027314) q[3];
cx q[2],q[3];
rz(0.10301430730992506) q[0];
rz(0.2365825665603178) q[1];
rz(0.1272869037136063) q[2];
rz(0.35477721191962563) q[3];
rx(-1.0370498864060722) q[0];
rx(-0.5791654010914185) q[1];
rx(-0.8401416713113637) q[2];
rx(-0.6101822746516505) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0980721892305274) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03610608472248012) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.22750619112891915) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2962549583471088) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2739933163274659) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.3265534602994102) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2391044268418144) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2679104244648102) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3469470641701602) q[3];
cx q[2],q[3];
rz(0.023063972389655706) q[0];
rz(0.10198284630123608) q[1];
rz(-0.06140885715019941) q[2];
rz(0.3392059109173982) q[3];
rx(-0.9998622380132614) q[0];
rx(-0.3390824399535341) q[1];
rx(-0.7613700607069055) q[2];
rx(-0.5911815500829529) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.12015593190716979) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.5268591688519828) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.3071234069387698) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.14507559844279377) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.03154667900185487) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.5193946495089268) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1537755768669734) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.12618754842920052) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3093793122385115) q[3];
cx q[2],q[3];
rz(-0.005029267448761878) q[0];
rz(0.18162732660369002) q[1];
rz(-0.06285021142819555) q[2];
rz(0.20282673349821997) q[3];
rx(-1.157731835013828) q[0];
rx(-0.5039655628948085) q[1];
rx(-0.47489338392956615) q[2];
rx(-0.7228924806550429) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.049011575507299766) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.4800518555901662) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.07060739168584301) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.167794011200185) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.08651236520367855) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.5274675574064758) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.14728605381660148) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07033011410421015) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.3091859491526369) q[3];
cx q[2],q[3];
rz(-0.18343051455349937) q[0];
rz(0.14149450853472473) q[1];
rz(0.024510128293103655) q[2];
rz(0.08850109475609202) q[3];
rx(-1.1265750370098944) q[0];
rx(-0.614989719501665) q[1];
rx(-0.42606987421314296) q[2];
rx(-0.7048466233119913) q[3];