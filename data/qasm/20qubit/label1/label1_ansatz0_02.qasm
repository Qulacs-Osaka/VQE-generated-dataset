OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.6519464531676845) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.76978070865181) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.01692759420010251) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.19414943275505678) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.1720823100577726) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.1315394167915217) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5683502009910595) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.3828127988268576) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.8736552175824053) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-1.1194144323493087) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.3723081918602986) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.6725918132060195) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.5060645588856253) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(1.6634647654021657) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(1.973305439935191) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(2.4142226961313837) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.02858051559918198) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.6057607560949113) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.5674274039582791) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.2423769450566676) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-1.4642753992131916) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.2640858664022261) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(2.132653514989954) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-1.0289275132379843) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.03689570347396459) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.011129281636499995) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-3.0557137930693754) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.9301553133809167) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.4627832119281465) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-1.591913067305565) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.041857982980476945) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.03166014491945931) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-1.3668703361953125) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(1.5382273334670362) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(0.03637654183633877) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(-0.6210529931777683) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(-0.7503441337119505) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(-1.477840654506948) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(1.5692617877147532) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(-3.1406091464217827) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(0.07881246429185544) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(0.014115788149122455) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(-0.035276441277708674) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(-1.7301720511428695) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(-1.5689366628386154) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(-0.07116676803400707) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(0.12601537098486576) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(0.5058239446202772) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(0.02904837509078402) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-0.01243304975615735) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(0.6754634922420173) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(-0.08391212209510004) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-0.09235784538666664) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(-1.852031511536206) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(0.37143598366116337) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(0.37428115926040034) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(0.34014730233142887) q[19];
cx q[18],q[19];
rz(-0.17024635690579104) q[0];
rz(-0.3596545729712699) q[1];
rz(0.08027110716260373) q[2];
rz(0.20516969103533367) q[3];
rz(-1.9674077420597276) q[4];
rz(-0.015265662153804507) q[5];
rz(1.5421981674146037) q[6];
rz(-0.0615438855308803) q[7];
rz(-1.4478578043040482) q[8];
rz(0.6364703363562911) q[9];
rz(-0.2025984903805271) q[10];
rz(-0.03113842592751223) q[11];
rz(-1.1741747309018462) q[12];
rz(0.09760870927440585) q[13];
rz(2.9704618924115276) q[14];
rz(-0.47918943021466565) q[15];
rz(-0.21944798236056803) q[16];
rz(-1.9885705362467452) q[17];
rz(1.198680761264167) q[18];
rz(2.79700481728191) q[19];
rx(-0.060920967204433095) q[0];
rx(-1.1215083379786766) q[1];
rx(-0.04500113193776203) q[2];
rx(-0.07179613044642177) q[3];
rx(-0.021852075623682705) q[4];
rx(-1.5804575405916337) q[5];
rx(1.563150998661002) q[6];
rx(-1.737976517635703) q[7];
rx(-1.5570647166391658) q[8];
rx(0.16210839991256243) q[9];
rx(0.9960138308482569) q[10];
rx(-1.6324060861346519) q[11];
rx(0.7608707345564679) q[12];
rx(-0.5263640278989686) q[13];
rx(0.9915795894779595) q[14];
rx(-0.25819872625426954) q[15];
rx(-1.1923483861711188) q[16];
rx(0.9945740660740394) q[17];
rx(-1.5775828461064436) q[18];
rx(0.6699505860139291) q[19];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.10942258037905823) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.6441181059123704) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.17672103681482088) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.2735187897186152) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.13815302030764912) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.15280009232123218) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5534788244156724) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.5987739307012598) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(2.034925631408559) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.2653035502561341) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.8383114499885028) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.18921716299213384) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(1.5189787837201991) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.003492346466765312) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.0027093179236411073) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.28129276921378726) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.6218165528776751) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.49107063432220643) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.29818120418648303) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-1.6020437327996233) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-1.5490045923763203) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.6976389989945989) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(1.7927491793293904) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(1.6144306584219792) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.006486552465010928) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-1.54193478829813) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(1.6799859641070602) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.15650077392710773) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.8289131960734715) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-1.770952759820381) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.013100814727868148) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.006806311901213773) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.0026718565875235986) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(-1.595065872785968) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(-0.03672629994361835) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(0.7111842482757145) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(0.004834291845218256) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(-1.3636724262735809) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(-0.6119277738255019) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(0.005761535958151692) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(0.009944543296692178) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(-0.07403346975018021) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(-0.19547710821379463) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(3.1402060051038028) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(-0.6388936019251024) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(-0.07550630392724639) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(2.759202896337551) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(0.07205641738173353) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(0.37171988849009213) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-0.3817360356230709) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(-0.9872137342347863) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(-3.120541596821829) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-1.5035031332971378) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(-0.019340066515377646) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(1.6267859384662613) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(-2.016713402062876) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(1.6188328496159938) q[19];
cx q[18],q[19];
rz(1.3547443629465197) q[0];
rz(-1.7580203729173693) q[1];
rz(-1.6334015210070694) q[2];
rz(-1.549955257318017) q[3];
rz(-0.6046152350597213) q[4];
rz(-0.06063439023753334) q[5];
rz(0.035302132238447) q[6];
rz(0.020391187931456896) q[7];
rz(0.25702721248122473) q[8];
rz(-3.1257186652412328) q[9];
rz(1.3545595891991145) q[10];
rz(1.468320430944184) q[11];
rz(0.16621381199902296) q[12];
rz(-0.09028872405544616) q[13];
rz(-0.30775128373560345) q[14];
rz(1.589702451263246) q[15];
rz(-0.9892265747227973) q[16];
rz(0.7770493077835539) q[17];
rz(-0.18485905119792884) q[18];
rz(1.617429868518324) q[19];
rx(-1.129129360842978) q[0];
rx(-1.1462820521111858) q[1];
rx(-0.3186720722276429) q[2];
rx(-1.116363669754088) q[3];
rx(0.38561686020531194) q[4];
rx(0.07100676916208401) q[5];
rx(0.2381818830940359) q[6];
rx(0.034836669261191416) q[7];
rx(-1.0040495412223054) q[8];
rx(-1.3967482637695654) q[9];
rx(-0.0677601906682089) q[10];
rx(-0.613498970910517) q[11];
rx(0.4871592053331473) q[12];
rx(0.06249412780005302) q[13];
rx(-0.1989899503759274) q[14];
rx(0.17686282323900807) q[15];
rx(-1.073731531012871) q[16];
rx(-0.030402394991065795) q[17];
rx(0.4804992357026983) q[18];
rx(-1.457473735788173) q[19];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.6660993906014268) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.802685979908061) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.24703107636370902) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.17280003819712084) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.14596555302504227) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0989932744704234) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.5214514615416179) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.017860017660928224) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.008507572762054131) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.9938720924671985) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.050088698139682676) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.03722174677443519) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.004833408044602902) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.01414041558373339) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.0060704495963061185) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.5617234336911378) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.6239000806262418) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.5178361104873347) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.3441166003587602) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.589734177522277) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.30879798803511943) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.08632586089639428) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.13885158852674817) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.12500656808204388) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(1.2961053506762241) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-1.3578085439911065) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.40248361113381703) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.1422169318247443) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(2.0499735898499782) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.1710423247792483) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-2.088292309295106) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(0.6904334510386141) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.7019357425034376) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(0.9809691456991776) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(0.6595899539481088) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(0.667002211007256) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(1.5195436992542444) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(1.1923148981003653) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(1.2056838465563033) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(-0.4636091210272921) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(0.7976263102361624) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(0.8175547846753957) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(-1.076155373724578) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(0.8522518176675815) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(0.8809830185045316) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(-0.41186585871955916) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(1.5859346556483183) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(0.0012537370501273823) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(-1.5577449203023388) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-0.47853664591844775) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(0.0032413233525073856) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(1.214112062194743) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-2.6280005855062702) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(0.8809151533510292) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(-0.5386286681717863) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(-0.027127478584464855) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(0.03493261566678105) q[19];
cx q[18],q[19];
rz(-1.1584577659053439) q[0];
rz(1.036430197453098) q[1];
rz(0.0550255323679854) q[2];
rz(-0.992601597864824) q[3];
rz(3.057925592494858) q[4];
rz(0.045878888191185234) q[5];
rz(-0.08523023647670526) q[6];
rz(-0.5559967992841279) q[7];
rz(-0.0027948477344090666) q[8];
rz(-0.09029819588555077) q[9];
rz(1.5660648643852553) q[10];
rz(1.5594294119786791) q[11];
rz(-1.5375471773398948) q[12];
rz(-1.5452291749718452) q[13];
rz(1.593824526129529) q[14];
rz(-0.7038162715561933) q[15];
rz(2.0176178765178174) q[16];
rz(-2.179505432253686) q[17];
rz(-2.056420985960658) q[18];
rz(-1.6097972450479379) q[19];
rx(-0.06646061163297896) q[0];
rx(0.3795684427447229) q[1];
rx(-2.778822164985101) q[2];
rx(1.0527952920347066) q[3];
rx(-1.1716306583491147) q[4];
rx(-0.0723091597819349) q[5];
rx(-0.06075248266247974) q[6];
rx(-0.15172725913845278) q[7];
rx(-0.22505090054056825) q[8];
rx(-1.9777289696487599) q[9];
rx(1.6008974445447508) q[10];
rx(-1.5274292834885568) q[11];
rx(-1.6636188867703396) q[12];
rx(1.536762903292147) q[13];
rx(-1.6030777296976761) q[14];
rx(-0.00278591002548522) q[15];
rx(3.1372834916558987) q[16];
rx(-0.9179300692108646) q[17];
rx(-0.9667161349153046) q[18];
rx(0.680633125025094) q[19];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.5409407979821999) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.6675479573383523) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(1.166852587422188) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.021766198726227943) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.014930157236909096) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.04413808997321788) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06242865734037335) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.19901076598907178) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.217562058704274) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.041791964812987345) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.028989483022084594) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.0856314891755235) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.5586132845686044) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-0.25755279810279524) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.496677516334351) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.9718278395523788) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.10550736881042053) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.8417432426955449) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.028626280973029335) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.016952474391424135) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.04469111249301303) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.06363229827432876) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.05238026263689908) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.1004389927111296) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(1.15800425601283) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.4613236199467311) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.8686746386555814) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.12297775349343305) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.11307414918269407) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.052442451035819924) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(0.12847741294155526) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.12139650257513503) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.3871953948129289) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(2.7329972696512566) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(0.40333664861243207) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(-0.011989705915616704) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(-0.15465907490072237) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(-0.14499864992435557) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(-0.05143764389751002) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(-0.10797505031150513) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(-0.13492313814862522) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(0.04037024201105556) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(-0.040677532838889315) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(0.011543232956343427) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(0.03391950690294736) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(-1.5314344956567711) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(-0.1376451570346636) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(-0.3076830043124151) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(0.024379975325823956) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(-0.008849864377498639) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(-0.13666998879595868) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(0.41935838968839023) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-1.7156630208321884) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(0.13333504168422225) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(0.20252973309064395) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(-0.24961288536981413) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(-0.011423909837615435) q[19];
cx q[18],q[19];
rz(-1.0032302994481432) q[0];
rz(-0.4445047233225666) q[1];
rz(-1.8324093805726938) q[2];
rz(-1.3177100417011653) q[3];
rz(0.0018694997466829756) q[4];
rz(0.014593812770369275) q[5];
rz(0.03466388680444434) q[6];
rz(0.4618101760620844) q[7];
rz(-0.18515362408062694) q[8];
rz(0.11886540772388304) q[9];
rz(0.2290129114923936) q[10];
rz(0.3364547752506374) q[11];
rz(-1.4279019389049334) q[12];
rz(-0.8999940667267625) q[13];
rz(0.42286873151326704) q[14];
rz(-1.9651145588532228) q[15];
rz(-0.9978231739723277) q[16];
rz(-0.08642295432470402) q[17];
rz(-1.6305695904851096) q[18];
rz(-3.1274026825757004) q[19];
rx(-0.669214430965211) q[0];
rx(-1.8138273197524175) q[1];
rx(-0.6894741575738381) q[2];
rx(-0.8453864356478757) q[3];
rx(-1.5729541107426863) q[4];
rx(-1.5758975857265036) q[5];
rx(1.5570727272972256) q[6];
rx(-1.4640953897409545) q[7];
rx(-1.6849965986300408) q[8];
rx(-0.03720765451098771) q[9];
rx(0.026804645936237728) q[10];
rx(-0.0038771412365490583) q[11];
rx(0.029336244759095863) q[12];
rx(-3.0971500002082033) q[13];
rx(-3.091831688380136) q[14];
rx(-0.007863771651553785) q[15];
rx(-3.113417458065676) q[16];
rx(-0.6158350050181025) q[17];
rx(-0.02296975520233779) q[18];
rx(-0.895291246825362) q[19];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.6383549213240102) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-2.2789381100079518) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.044220692060786965) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.11717843297278621) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.13635108571371782) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.024114455150593926) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2712024956381628) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1996958008565432) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1013754502864314) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.07880095302115811) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.07119981418875854) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.1258799785103898) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(0.2760464253300475) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.2479462443966667) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.0632522155414013) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-1.7546178841886384) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.5910934903461793) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.32733206636004925) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.4274211321426097) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.41456897202366993) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.44698832499310925) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.31501010279910746) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.3593208759782126) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.2551986442644388) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.606082758536384) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.5356712984128857) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.335959842659874) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.1635322038668205) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.16138022033327565) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.09398017008413015) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-0.5025650413751688) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.5014219012410288) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.0714353346613653) q[11];
cx q[10],q[11];
h q[11];
h q[12];
cx q[11],q[12];
rz(-2.5643056198953587) q[12];
cx q[11],q[12];
h q[11];
h q[12];
sdg q[11];
h q[11];
sdg q[12];
h q[12];
cx q[11],q[12];
rz(-0.5600461472259028) q[12];
cx q[11],q[12];
h q[11];
s q[11];
h q[12];
s q[12];
cx q[11],q[12];
rz(0.026067160658658675) q[12];
cx q[11],q[12];
h q[12];
h q[13];
cx q[12],q[13];
rz(0.03547725387369994) q[13];
cx q[12],q[13];
h q[12];
h q[13];
sdg q[12];
h q[12];
sdg q[13];
h q[13];
cx q[12],q[13];
rz(0.03942656582495279) q[13];
cx q[12],q[13];
h q[12];
s q[12];
h q[13];
s q[13];
cx q[12],q[13];
rz(-0.14091267543662217) q[13];
cx q[12],q[13];
h q[13];
h q[14];
cx q[13],q[14];
rz(-0.5236625371202148) q[14];
cx q[13],q[14];
h q[13];
h q[14];
sdg q[13];
h q[13];
sdg q[14];
h q[14];
cx q[13],q[14];
rz(-0.4815475523488989) q[14];
cx q[13],q[14];
h q[13];
s q[13];
h q[14];
s q[14];
cx q[13],q[14];
rz(-0.3007689719459756) q[14];
cx q[13],q[14];
h q[14];
h q[15];
cx q[14],q[15];
rz(0.25941721799722783) q[15];
cx q[14],q[15];
h q[14];
h q[15];
sdg q[14];
h q[14];
sdg q[15];
h q[15];
cx q[14],q[15];
rz(-0.2314152578496393) q[15];
cx q[14],q[15];
h q[14];
s q[14];
h q[15];
s q[15];
cx q[14],q[15];
rz(-0.2893422891227417) q[15];
cx q[14],q[15];
h q[15];
h q[16];
cx q[15],q[16];
rz(-0.954680641287521) q[16];
cx q[15],q[16];
h q[15];
h q[16];
sdg q[15];
h q[15];
sdg q[16];
h q[16];
cx q[15],q[16];
rz(2.2537189416104804) q[16];
cx q[15],q[16];
h q[15];
s q[15];
h q[16];
s q[16];
cx q[15],q[16];
rz(0.2275238186555603) q[16];
cx q[15],q[16];
h q[16];
h q[17];
cx q[16],q[17];
rz(0.384148872471626) q[17];
cx q[16],q[17];
h q[16];
h q[17];
sdg q[16];
h q[16];
sdg q[17];
h q[17];
cx q[16],q[17];
rz(0.38501727231557054) q[17];
cx q[16],q[17];
h q[16];
s q[16];
h q[17];
s q[17];
cx q[16],q[17];
rz(0.34579000646064856) q[17];
cx q[16],q[17];
h q[17];
h q[18];
cx q[17],q[18];
rz(0.7788290424853413) q[18];
cx q[17],q[18];
h q[17];
h q[18];
sdg q[17];
h q[17];
sdg q[18];
h q[18];
cx q[17],q[18];
rz(-0.709014734027288) q[18];
cx q[17],q[18];
h q[17];
s q[17];
h q[18];
s q[18];
cx q[17],q[18];
rz(-0.6350781613178295) q[18];
cx q[17],q[18];
h q[18];
h q[19];
cx q[18],q[19];
rz(0.618019246832174) q[19];
cx q[18],q[19];
h q[18];
h q[19];
sdg q[18];
h q[18];
sdg q[19];
h q[19];
cx q[18],q[19];
rz(-0.6205463621916728) q[19];
cx q[18],q[19];
h q[18];
s q[18];
h q[19];
s q[19];
cx q[18],q[19];
rz(-0.5480745989911212) q[19];
cx q[18],q[19];
rz(0.2523682630044424) q[0];
rz(1.652788213523674) q[1];
rz(3.0540672511067153) q[2];
rz(2.3111546010487825) q[3];
rz(-0.8363202297677198) q[4];
rz(-1.3769190521360648) q[5];
rz(-1.282662727071682) q[6];
rz(1.252737961788521) q[7];
rz(1.344209508962809) q[8];
rz(1.4007250867051797) q[9];
rz(0.9045249480809477) q[10];
rz(-1.0700172537120338) q[11];
rz(-2.8031990157854536) q[12];
rz(-1.555714483179824) q[13];
rz(-1.013181561999307) q[14];
rz(3.0448624454330235) q[15];
rz(-0.2571577999921971) q[16];
rz(0.3869076828603858) q[17];
rz(-0.609640430648399) q[18];
rz(1.9350153039692664) q[19];
rx(-0.5065688126831743) q[0];
rx(-0.03982656726441036) q[1];
rx(-0.6061490926774123) q[2];
rx(-3.0646419840218413) q[3];
rx(3.136139430589502) q[4];
rx(-3.126943952273512) q[5];
rx(3.099236019728691) q[6];
rx(-0.04676279254833087) q[7];
rx(0.027441378169902705) q[8];
rx(0.03151301760232772) q[9];
rx(0.000900732773329812) q[10];
rx(-3.1263829608461897) q[11];
rx(0.025312905411795133) q[12];
rx(0.012841689910376736) q[13];
rx(-0.005058172418509324) q[14];
rx(-0.0045722829860653175) q[15];
rx(0.003268254795438216) q[16];
rx(-0.009170856327103037) q[17];
rx(-3.136594575107594) q[18];
rx(-0.01363439567778533) q[19];