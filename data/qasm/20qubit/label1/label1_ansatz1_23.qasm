OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.964638507164119) q[0];
rz(2.8900872765210415) q[0];
ry(0.8438334724616421) q[1];
rz(-1.2534403542617303) q[1];
ry(0.804528426563925) q[2];
rz(1.3134170503326905) q[2];
ry(-0.02411537744990988) q[3];
rz(0.9908276936436151) q[3];
ry(-2.4512523126208725) q[4];
rz(-0.058440789039416075) q[4];
ry(-1.6506968337467272) q[5];
rz(-0.5853722971771594) q[5];
ry(2.6893440004313245) q[6];
rz(-1.6365276179925088) q[6];
ry(-0.21228260802273627) q[7];
rz(2.089616045479234) q[7];
ry(0.7336054001533086) q[8];
rz(-1.2392536890999364) q[8];
ry(1.376329951669247) q[9];
rz(-2.63996871704756) q[9];
ry(-2.870014766007092) q[10];
rz(-0.5687581940938155) q[10];
ry(2.022553855142084) q[11];
rz(0.24578301979125694) q[11];
ry(-0.7162055854945973) q[12];
rz(2.7945130485693075) q[12];
ry(-2.8891855150094297) q[13];
rz(-2.0758501970580787) q[13];
ry(1.7949298800102262) q[14];
rz(2.5424083164118847) q[14];
ry(0.7086309548002704) q[15];
rz(-2.521310424626649) q[15];
ry(0.7512882613474784) q[16];
rz(0.48295723333062573) q[16];
ry(3.1266090064212655) q[17];
rz(1.3615947763098686) q[17];
ry(-3.1241440767958837) q[18];
rz(-0.44186414661501416) q[18];
ry(-2.8779385994042053) q[19];
rz(-2.5442505908000785) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.703393023816329) q[0];
rz(0.12283020359892306) q[0];
ry(-2.451500484319991) q[1];
rz(0.37102292535452275) q[1];
ry(-0.9013222921560029) q[2];
rz(-1.9738542993669748) q[2];
ry(-1.115411927574614) q[3];
rz(-1.2111460545469885) q[3];
ry(2.955884702077034) q[4];
rz(-1.4547807533418873) q[4];
ry(2.9841820775444114) q[5];
rz(-1.0691851300678437) q[5];
ry(2.7605448537919597) q[6];
rz(2.0852738300443594) q[6];
ry(-3.0509274173382916) q[7];
rz(0.2749150011004092) q[7];
ry(0.2619293556394817) q[8];
rz(-0.7666496288578871) q[8];
ry(0.3193583846130279) q[9];
rz(0.44930592181835927) q[9];
ry(0.031895252410016894) q[10];
rz(0.33795351436765064) q[10];
ry(3.1149754217348615) q[11];
rz(2.0377693363833185) q[11];
ry(0.0049733352279088695) q[12];
rz(1.5129284829002865) q[12];
ry(0.7948065702222621) q[13];
rz(-0.24403354626432489) q[13];
ry(-2.408611024558155) q[14];
rz(-0.18106865179549386) q[14];
ry(0.19588375314699924) q[15];
rz(-3.1380416429351006) q[15];
ry(-1.1688223714333565) q[16];
rz(0.9918772207735501) q[16];
ry(-2.558648330147619) q[17];
rz(2.433676458268666) q[17];
ry(0.04260586066105426) q[18];
rz(1.3842792495910043) q[18];
ry(1.3754839326522603) q[19];
rz(-0.2157772486349319) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.617139075324082) q[0];
rz(1.6539583813654408) q[0];
ry(1.4693543635024033) q[1];
rz(2.253471369221439) q[1];
ry(-0.0002701026816858132) q[2];
rz(-1.8448344270308048) q[2];
ry(-3.123811767747874) q[3];
rz(1.9304951701365614) q[3];
ry(-3.140102085306457) q[4];
rz(0.7714764585082136) q[4];
ry(-2.94050834652694) q[5];
rz(2.307866657019346) q[5];
ry(3.0029902167424125) q[6];
rz(3.0236549663157204) q[6];
ry(0.3241859088717538) q[7];
rz(-0.7302392736979036) q[7];
ry(-2.0231261020306834) q[8];
rz(1.9809574046858938) q[8];
ry(-3.0171088711500724) q[9];
rz(-0.9852570729469278) q[9];
ry(-2.8014154365329844) q[10];
rz(-1.8188112060112323) q[10];
ry(1.0312511825742494) q[11];
rz(-1.9674004041664945) q[11];
ry(1.1005439099560244) q[12];
rz(-0.5837849875618702) q[12];
ry(-1.3100055932454227) q[13];
rz(2.7230880905971366) q[13];
ry(3.1243622421735413) q[14];
rz(-2.8814015467022878) q[14];
ry(0.6606096954326991) q[15];
rz(-2.3005295713351708) q[15];
ry(-0.07275508440079115) q[16];
rz(-1.9722597265539479) q[16];
ry(-0.023952824160655695) q[17];
rz(-2.948513627120927) q[17];
ry(0.4393103920798449) q[18];
rz(-1.460260844009409) q[18];
ry(2.546243957941103) q[19];
rz(2.8727077303937825) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.7347460539152104) q[0];
rz(2.8890140845387235) q[0];
ry(0.5549048145300883) q[1];
rz(-0.6195170363113363) q[1];
ry(2.0613786679285795) q[2];
rz(2.064035478914705) q[2];
ry(-2.0187586241204554) q[3];
rz(2.8179829132170493) q[3];
ry(0.27982503986313206) q[4];
rz(0.41999308981204136) q[4];
ry(0.5935366985248143) q[5];
rz(0.6388824719272196) q[5];
ry(3.0658577470856763) q[6];
rz(1.120826428809992) q[6];
ry(-2.4499986261400024) q[7];
rz(-0.5284791560586383) q[7];
ry(1.1556439367497702) q[8];
rz(-2.3364881769729884) q[8];
ry(-2.596185079489218) q[9];
rz(-2.116620928189638) q[9];
ry(1.9978381120357565) q[10];
rz(1.6798883715787714) q[10];
ry(1.6317994510133202) q[11];
rz(0.809368930961982) q[11];
ry(-3.131749981070252) q[12];
rz(-0.28782642377001216) q[12];
ry(2.7998699764646786) q[13];
rz(2.5081640224456625) q[13];
ry(0.6986878272062399) q[14];
rz(-1.9493535151617971) q[14];
ry(0.26700183473748723) q[15];
rz(-1.8925243503863785) q[15];
ry(0.37017954405708287) q[16];
rz(-2.6979679735081477) q[16];
ry(-0.6296632124112351) q[17];
rz(1.6095791868941467) q[17];
ry(-3.0412320854654) q[18];
rz(-2.0146653502166174) q[18];
ry(0.3631460425567165) q[19];
rz(-0.2081839895938922) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.390088040417696) q[0];
rz(1.7405041817810742) q[0];
ry(-0.8522204438584513) q[1];
rz(2.7604826710915593) q[1];
ry(2.9161018199403905) q[2];
rz(2.836235352828473) q[2];
ry(0.015329983035515207) q[3];
rz(0.1268776203517215) q[3];
ry(0.9993233622387229) q[4];
rz(-2.0891862573938376) q[4];
ry(-1.4777481518874795) q[5];
rz(1.0822178136754237) q[5];
ry(-3.1025619215083484) q[6];
rz(1.662863559942217) q[6];
ry(0.10431801017096004) q[7];
rz(2.5874106564580903) q[7];
ry(3.0073801866543355) q[8];
rz(0.7528018530086247) q[8];
ry(-1.028861496528772) q[9];
rz(2.9137490750552657) q[9];
ry(2.4115657216230266) q[10];
rz(2.979165045806434) q[10];
ry(3.104630565130785) q[11];
rz(-0.26416961318564347) q[11];
ry(-2.1362555155659857) q[12];
rz(-2.5834343662848167) q[12];
ry(-0.9770591027068578) q[13];
rz(-1.1192850993793895) q[13];
ry(-1.4792286466241567) q[14];
rz(1.2922080303429722) q[14];
ry(-0.4768331237437895) q[15];
rz(1.675558291638486) q[15];
ry(0.018626673221422148) q[16];
rz(3.0319368519776067) q[16];
ry(-0.016319419542497285) q[17];
rz(1.1928946788943877) q[17];
ry(0.5480397329793582) q[18];
rz(-1.2665587380874088) q[18];
ry(-2.638985579133898) q[19];
rz(-2.259597609919589) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.202858193921831) q[0];
rz(0.5461027655247034) q[0];
ry(0.3486178219393439) q[1];
rz(2.891342928915632) q[1];
ry(-2.3489572241402827) q[2];
rz(-1.5453460366660288) q[2];
ry(-3.138649883779884) q[3];
rz(2.785953069252726) q[3];
ry(-2.9031674633442286) q[4];
rz(-1.9212111791933213) q[4];
ry(-0.6971658279293802) q[5];
rz(2.986794716751686) q[5];
ry(0.11115668040062843) q[6];
rz(1.1707891811853024) q[6];
ry(0.15918453650412445) q[7];
rz(-2.8207532572645637) q[7];
ry(-0.4223047415154175) q[8];
rz(-2.9642144219901403) q[8];
ry(-1.061302529593577) q[9];
rz(-0.01569033027240826) q[9];
ry(-3.0495700399105683) q[10];
rz(-1.6506154055730073) q[10];
ry(3.1336492581240996) q[11];
rz(3.1280191384879608) q[11];
ry(0.013993297777243006) q[12];
rz(2.5540024080014843) q[12];
ry(3.1276541136448066) q[13];
rz(0.041499230182902906) q[13];
ry(2.172519983399721) q[14];
rz(-2.1519555987058103) q[14];
ry(-2.5727668457025445) q[15];
rz(-0.5385271076504994) q[15];
ry(-1.9038019449074775) q[16];
rz(-1.3794478979719775) q[16];
ry(-3.1079544345645007) q[17];
rz(2.9525786263902174) q[17];
ry(-3.042180672647511) q[18];
rz(-2.9111389847215476) q[18];
ry(-0.9447350738318098) q[19];
rz(0.14391956819352547) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.47031282866490454) q[0];
rz(-1.5115139544791125) q[0];
ry(0.13131988404611405) q[1];
rz(-2.068821779225442) q[1];
ry(0.053159790622213876) q[2];
rz(-2.6413909095172183) q[2];
ry(3.0734186786745274) q[3];
rz(-0.03966361513898953) q[3];
ry(1.6182760787695836) q[4];
rz(-3.0873065986759234) q[4];
ry(-1.7085121995226658) q[5];
rz(1.156753601326359) q[5];
ry(-3.1031004586827553) q[6];
rz(0.03411287744309751) q[6];
ry(2.94345882430171) q[7];
rz(-0.9511100935635449) q[7];
ry(0.0003892678746558358) q[8];
rz(2.895678691048104) q[8];
ry(1.2173835200354066) q[9];
rz(-2.024012408600807) q[9];
ry(-1.6759539750590733) q[10];
rz(-1.786284637693961) q[10];
ry(1.1189899904679674) q[11];
rz(2.863014285254952) q[11];
ry(-0.9889969202334505) q[12];
rz(-0.840324757236727) q[12];
ry(0.9994862653948259) q[13];
rz(0.9401184978997463) q[13];
ry(1.8243519986210353) q[14];
rz(0.48523460815344815) q[14];
ry(0.3313904474714331) q[15];
rz(-3.0941578089911004) q[15];
ry(0.021328438597702108) q[16];
rz(-1.1010118146537613) q[16];
ry(-3.0890280411645312) q[17];
rz(-1.3423968823561427) q[17];
ry(0.5108879640920398) q[18];
rz(2.5683924361903236) q[18];
ry(-2.783976200137452) q[19];
rz(2.7067270269440145) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.0048950442243498316) q[0];
rz(-1.0749324753197511) q[0];
ry(-0.018583182912594778) q[1];
rz(-0.28882912670642563) q[1];
ry(-3.1064264152719105) q[2];
rz(0.13555131499268894) q[2];
ry(2.053218740509575) q[3];
rz(0.022153836342928557) q[3];
ry(-2.6612263389732473) q[4];
rz(1.7588453211617283) q[4];
ry(0.22659254867450887) q[5];
rz(2.3030943194892872) q[5];
ry(2.9530191683379057) q[6];
rz(2.440919134926087) q[6];
ry(-2.537680970636607) q[7];
rz(1.682406779696743) q[7];
ry(1.9408398596255463) q[8];
rz(1.6110226096298723) q[8];
ry(-0.957656720345688) q[9];
rz(-2.6702855454740644) q[9];
ry(2.7166423499430508) q[10];
rz(-2.817265451477242) q[10];
ry(0.29980069807180243) q[11];
rz(-0.860323646061711) q[11];
ry(-0.016758758301261167) q[12];
rz(-0.7780302397960224) q[12];
ry(-2.448666640357045) q[13];
rz(0.1806322773896354) q[13];
ry(1.388330396138029) q[14];
rz(0.7313151129267622) q[14];
ry(3.0335909083925743) q[15];
rz(-1.725232109334692) q[15];
ry(2.2688308586951385) q[16];
rz(2.9805841103760256) q[16];
ry(-0.2950183320855384) q[17];
rz(1.21352106878707) q[17];
ry(2.999777753002209) q[18];
rz(-2.9258691403849966) q[18];
ry(-2.609373851615149) q[19];
rz(-1.7249514346921675) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.1041386701809004) q[0];
rz(-0.47069961584896985) q[0];
ry(2.7616927534910367) q[1];
rz(2.688858795753654) q[1];
ry(3.1409240459488323) q[2];
rz(-3.0194089300163696) q[2];
ry(-0.8310702309444772) q[3];
rz(-0.00015098172570882326) q[3];
ry(-0.024907367016931303) q[4];
rz(-0.40572531422218105) q[4];
ry(-2.3865696503332408) q[5];
rz(-0.46830081844591587) q[5];
ry(1.8262444905093222) q[6];
rz(-2.931281099447944) q[6];
ry(-0.6192074843635631) q[7];
rz(0.23363678182583844) q[7];
ry(-1.903763088067576) q[8];
rz(-0.6906952827061801) q[8];
ry(3.0804929238155934) q[9];
rz(-0.24035844944146592) q[9];
ry(1.0108735779715001) q[10];
rz(-0.24410714743153195) q[10];
ry(-2.350643654952729) q[11];
rz(2.590774608918421) q[11];
ry(0.03147788311878184) q[12];
rz(2.655523727120254) q[12];
ry(1.2167030487429251) q[13];
rz(0.18769907898531135) q[13];
ry(-2.693947873333387) q[14];
rz(1.5087698406965169) q[14];
ry(0.6133608294216043) q[15];
rz(0.6153761577961879) q[15];
ry(-0.046477151384168545) q[16];
rz(-0.692123233684027) q[16];
ry(-0.056000683666828095) q[17];
rz(0.6599545583578474) q[17];
ry(1.2731100309792902) q[18];
rz(-0.8047109212676942) q[18];
ry(0.4648000242352317) q[19];
rz(1.194942690111037) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.6525710650898997) q[0];
rz(-2.7267331400192543) q[0];
ry(0.36211170857914876) q[1];
rz(-2.148822805954111) q[1];
ry(0.1336139279226943) q[2];
rz(0.09482520349844313) q[2];
ry(-1.0837007189088759) q[3];
rz(1.1997312072341375) q[3];
ry(3.0785100816648154) q[4];
rz(1.0052584744133362) q[4];
ry(-0.3020369162598829) q[5];
rz(-2.805185302638897) q[5];
ry(-2.625410715391097) q[6];
rz(0.1766815879484227) q[6];
ry(-0.5472377496857712) q[7];
rz(-2.4400921811148355) q[7];
ry(-0.9093058033386523) q[8];
rz(2.6967820311011024) q[8];
ry(2.4827314654818986) q[9];
rz(2.2112234780221085) q[9];
ry(-0.026726431793558218) q[10];
rz(-2.5282058184386575) q[10];
ry(-2.915564275853381) q[11];
rz(-2.4664327222388995) q[11];
ry(-3.1397710259519545) q[12];
rz(2.4330649961056197) q[12];
ry(1.619090834125825) q[13];
rz(2.0679686108901416) q[13];
ry(-0.4256346664715922) q[14];
rz(1.7222720572826358) q[14];
ry(-2.671851502589541) q[15];
rz(1.2812675583519002) q[15];
ry(-0.9305275453753064) q[16];
rz(0.32503809062335975) q[16];
ry(-2.8587719395572964) q[17];
rz(1.3231374714997424) q[17];
ry(-0.1205619094500072) q[18];
rz(-1.0909465197291688) q[18];
ry(1.9799379282487033) q[19];
rz(1.2095648821984053) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.9123207683173922) q[0];
rz(1.3959855912638042) q[0];
ry(3.092211248054617) q[1];
rz(2.9212244154726035) q[1];
ry(-1.3373075640402288) q[2];
rz(1.4200826838804959) q[2];
ry(-3.054876016122473) q[3];
rz(2.408027863327781) q[3];
ry(-0.8832946138717378) q[4];
rz(2.5225941580680633) q[4];
ry(-2.5511333243016834) q[5];
rz(-1.4863365441119791) q[5];
ry(2.6769394114502343) q[6];
rz(-0.5129523123646408) q[6];
ry(2.020931536096274) q[7];
rz(-0.057103824299330044) q[7];
ry(3.1114200384728496) q[8];
rz(0.6560794273119499) q[8];
ry(2.0325696584091055) q[9];
rz(-0.07618146358456636) q[9];
ry(0.6438003087491355) q[10];
rz(2.298807619505645) q[10];
ry(-1.858539070570595) q[11];
rz(-1.7214527867615361) q[11];
ry(3.1205851986697004) q[12];
rz(-1.6050017675441364) q[12];
ry(2.9370167922634667) q[13];
rz(-1.7040932664536426) q[13];
ry(-2.100266793964141) q[14];
rz(-2.1063061823253806) q[14];
ry(2.437983309463287) q[15];
rz(-0.9621901304842981) q[15];
ry(0.0465087550271397) q[16];
rz(2.191322692607906) q[16];
ry(-0.343805443525747) q[17];
rz(0.4541403719069192) q[17];
ry(0.6612430614118343) q[18];
rz(0.16255826316285907) q[18];
ry(1.3332475342345331) q[19];
rz(0.9588616417402307) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.6107660090468698) q[0];
rz(-1.5782622999519083) q[0];
ry(2.950652494993221) q[1];
rz(-2.0852574508949253) q[1];
ry(-1.1762443215486318) q[2];
rz(-0.522302766200245) q[2];
ry(-3.1002968075165933) q[3];
rz(3.0357456113688728) q[3];
ry(3.097662072509648) q[4];
rz(2.326459069521007) q[4];
ry(0.9019395356890172) q[5];
rz(-1.7136535628940743) q[5];
ry(1.32652621561515) q[6];
rz(-0.2254561887322931) q[6];
ry(-2.4506475796179794) q[7];
rz(2.6633200676256985) q[7];
ry(-0.715343855249458) q[8];
rz(1.0039842866783357) q[8];
ry(1.7841279360946434) q[9];
rz(-2.4110563583338136) q[9];
ry(-1.3938088759325193) q[10];
rz(-3.12299046237482) q[10];
ry(-0.7242500156286376) q[11];
rz(-0.7239239232397474) q[11];
ry(3.137349667753259) q[12];
rz(-1.593323579516401) q[12];
ry(0.420788263079599) q[13];
rz(2.075666313442048) q[13];
ry(-2.023750407218334) q[14];
rz(0.11691627586005923) q[14];
ry(0.007641202455125118) q[15];
rz(2.2532912873439925) q[15];
ry(0.1533454544355095) q[16];
rz(-1.16749665343269) q[16];
ry(-0.2002836028766136) q[17];
rz(-2.3654774867184587) q[17];
ry(0.4013942621642762) q[18];
rz(2.588307997135665) q[18];
ry(1.2381247926993026) q[19];
rz(0.6779260335014425) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.2882061652629657) q[0];
rz(-1.3439890928577365) q[0];
ry(1.927255736698612) q[1];
rz(-1.7084374450529167) q[1];
ry(-1.4030647465817294) q[2];
rz(-2.450416387038539) q[2];
ry(-0.31745863883833775) q[3];
rz(-1.7476036493887275) q[3];
ry(-1.3288687544685578) q[4];
rz(-2.3804207626708846) q[4];
ry(-2.3819791091786437) q[5];
rz(-1.4598749000185098) q[5];
ry(2.9551070113347824) q[6];
rz(3.0879644336183287) q[6];
ry(-1.9199813879013103) q[7];
rz(-1.834869647339354) q[7];
ry(1.1823106277259186) q[8];
rz(-1.1746726402315475) q[8];
ry(-3.0761100607552723) q[9];
rz(3.090897192007076) q[9];
ry(1.5801421315761317) q[10];
rz(0.7044091723635127) q[10];
ry(-1.6640390904018008) q[11];
rz(-1.2177886339776236) q[11];
ry(-2.0120764485738736) q[12];
rz(-0.09310771162199762) q[12];
ry(2.8398018582955027) q[13];
rz(0.5099215948802945) q[13];
ry(-1.8457340612015196) q[14];
rz(-2.9531871423062985) q[14];
ry(0.12979270280799785) q[15];
rz(0.3696331306310805) q[15];
ry(-3.1332799456434457) q[16];
rz(1.0877571406490847) q[16];
ry(0.6360820425028187) q[17];
rz(-1.8567294567203891) q[17];
ry(0.1301322237304374) q[18];
rz(1.0591007339131608) q[18];
ry(-2.867358532070382) q[19];
rz(-0.0004965911651684253) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.30880091749095834) q[0];
rz(0.09465191106182483) q[0];
ry(-3.001608016674349) q[1];
rz(-0.41520262785914497) q[1];
ry(-3.1167237926313716) q[2];
rz(0.7258733955667002) q[2];
ry(3.0895046525133254) q[3];
rz(0.3204847079622644) q[3];
ry(-3.1386190981017665) q[4];
rz(1.8925004036015949) q[4];
ry(0.031410773301587376) q[5];
rz(-0.5498724063776006) q[5];
ry(2.4375221259634277) q[6];
rz(-1.126610133578787) q[6];
ry(-0.12084104206994885) q[7];
rz(1.7786239422967647) q[7];
ry(-0.09057794930875163) q[8];
rz(-2.5109543385045225) q[8];
ry(-0.02337408697364177) q[9];
rz(0.8165944799176197) q[9];
ry(2.460660508267635) q[10];
rz(-1.1233567067630865) q[10];
ry(0.01707210164446193) q[11];
rz(1.3235563622144255) q[11];
ry(-3.134471594716477) q[12];
rz(-0.04212301082766554) q[12];
ry(-2.1817750147891166) q[13];
rz(2.0025677653170773) q[13];
ry(1.8885672882546718) q[14];
rz(1.9063481080694291) q[14];
ry(2.860751451602437) q[15];
rz(3.0448623272505513) q[15];
ry(0.8070539144311688) q[16];
rz(1.0282141420215822) q[16];
ry(-3.0539945717352546) q[17];
rz(-0.9059442320943939) q[17];
ry(-0.9084388186072294) q[18];
rz(-1.5634755685694477) q[18];
ry(-2.697600073065528) q[19];
rz(0.5804021254073808) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.1077417486370662) q[0];
rz(-2.687122611536481) q[0];
ry(2.941961573820819) q[1];
rz(3.115382557415451) q[1];
ry(1.1142157942707627) q[2];
rz(-2.278278967323173) q[2];
ry(-1.6830616974268278) q[3];
rz(0.5420308496275023) q[3];
ry(1.2016657137726063) q[4];
rz(-0.7759296555807748) q[4];
ry(2.035816216285454) q[5];
rz(-0.9613119167380272) q[5];
ry(-0.029703110023030273) q[6];
rz(1.1405887470633065) q[6];
ry(1.055495821705873) q[7];
rz(0.03415098727442967) q[7];
ry(-1.660796009081972) q[8];
rz(1.7210278779877257) q[8];
ry(-3.130633095985337) q[9];
rz(2.3867851682010346) q[9];
ry(-0.19924203790756678) q[10];
rz(2.443490515451838) q[10];
ry(2.3540744998683465) q[11];
rz(0.32535483686974764) q[11];
ry(0.8572134711224955) q[12];
rz(-1.3486197559683832) q[12];
ry(3.121711267646749) q[13];
rz(2.0225119496021753) q[13];
ry(0.1983886590954187) q[14];
rz(-3.092978073238881) q[14];
ry(3.09570994517847) q[15];
rz(0.7696114895061041) q[15];
ry(-0.029590215562461886) q[16];
rz(-0.752069132269618) q[16];
ry(0.25749681342378766) q[17];
rz(1.7838420560505792) q[17];
ry(-3.1222596958253543) q[18];
rz(0.3824645044689984) q[18];
ry(-2.5289665528268515) q[19];
rz(-0.8243367874989298) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.5448559851433332) q[0];
rz(-2.434175666620632) q[0];
ry(1.5377420491592497) q[1];
rz(1.748321838073025) q[1];
ry(-3.0998154775928857) q[2];
rz(0.5703447542028359) q[2];
ry(-3.123018649804466) q[3];
rz(-1.3755525945609302) q[3];
ry(0.8665047356041812) q[4];
rz(-2.9921460457029907) q[4];
ry(3.1333075476405106) q[5];
rz(-0.5075729649733753) q[5];
ry(-1.2679335969207428) q[6];
rz(3.1085752977598387) q[6];
ry(-1.8081489492911342) q[7];
rz(-0.0046925124802337246) q[7];
ry(-0.6934059550767779) q[8];
rz(0.6313218458547666) q[8];
ry(3.122600376998268) q[9];
rz(-1.7911089195085719) q[9];
ry(-1.062281317407567) q[10];
rz(2.573776524698378) q[10];
ry(-0.4276236655416783) q[11];
rz(-1.093702488375786) q[11];
ry(-0.01610672192876894) q[12];
rz(0.4739650750518631) q[12];
ry(-1.510851459679568) q[13];
rz(-2.598933931852066) q[13];
ry(0.5137754371557772) q[14];
rz(-2.3129527215820707) q[14];
ry(-0.13838881140175163) q[15];
rz(0.7565611622579529) q[15];
ry(-2.4626056050075666) q[16];
rz(2.9871541842918305) q[16];
ry(0.46513880687777726) q[17];
rz(-1.058422217344826) q[17];
ry(-0.41918136458964483) q[18];
rz(-3.137901252600492) q[18];
ry(1.6285603797013517) q[19];
rz(1.8536891240258626) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.5974464268452175) q[0];
rz(-1.1014534121481052) q[0];
ry(-0.700142448513572) q[1];
rz(0.8165174204811052) q[1];
ry(1.6256722657959441) q[2];
rz(-3.01420642458769) q[2];
ry(-1.2890404393902692) q[3];
rz(0.11138123443551862) q[3];
ry(0.8872503208666656) q[4];
rz(2.8494838619286056) q[4];
ry(-0.23149758416096375) q[5];
rz(2.9696984621419267) q[5];
ry(-1.9284990779759106) q[6];
rz(1.383819942160955) q[6];
ry(-2.906861550797329) q[7];
rz(2.8132311369292657) q[7];
ry(-0.16290552967814076) q[8];
rz(-1.23835511198783) q[8];
ry(-0.04332747731220371) q[9];
rz(-3.0541972203122203) q[9];
ry(2.8309088684111448) q[10];
rz(2.7089648186447524) q[10];
ry(0.7396685357313091) q[11];
rz(0.053416545822804085) q[11];
ry(2.346240998684087) q[12];
rz(-2.323189015446433) q[12];
ry(-2.6146498470847632) q[13];
rz(-1.4527721311502415) q[13];
ry(0.6252327034034167) q[14];
rz(2.7159616588954156) q[14];
ry(-1.9312770779192199) q[15];
rz(-1.4566528005120647) q[15];
ry(-3.1296552828271347) q[16];
rz(2.189618642842701) q[16];
ry(2.053311118539421) q[17];
rz(-1.4138368440183477) q[17];
ry(3.1116577482850807) q[18];
rz(-1.7972823464967131) q[18];
ry(-0.2953523555844084) q[19];
rz(2.3141881253707233) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.7057592390849887) q[0];
rz(-0.6140910129643968) q[0];
ry(-0.07426912617826353) q[1];
rz(-2.8884799198385864) q[1];
ry(1.5937265408715795) q[2];
rz(0.9458614333249626) q[2];
ry(-1.5938700081297217) q[3];
rz(-2.5074064425671856) q[3];
ry(2.260609292030339) q[4];
rz(2.674077518835174) q[4];
ry(2.5382432822853733) q[5];
rz(-0.016775552208036698) q[5];
ry(-2.997049614180732) q[6];
rz(-2.843811117958531) q[6];
ry(2.424365803360663) q[7];
rz(-3.056875635834162) q[7];
ry(-1.1359368353862278) q[8];
rz(-2.214510212282978) q[8];
ry(3.0207621046840964) q[9];
rz(0.45334798135322973) q[9];
ry(-2.793488190293054) q[10];
rz(0.7866409391473718) q[10];
ry(-0.0010413719659636556) q[11];
rz(0.602021302312596) q[11];
ry(-3.1379521400033865) q[12];
rz(1.7224208397046186) q[12];
ry(1.5797164809705293) q[13];
rz(-2.1092399485587614) q[13];
ry(0.0314856223964457) q[14];
rz(1.7726983145377107) q[14];
ry(-3.086878213222573) q[15];
rz(1.736097541525785) q[15];
ry(0.014079532027306387) q[16];
rz(1.5619213723705019) q[16];
ry(-0.6409360375142678) q[17];
rz(-2.4908520634273374) q[17];
ry(-2.2354252162019703) q[18];
rz(1.842977000129947) q[18];
ry(1.6192857602834785) q[19];
rz(2.965484037860985) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.680623053184722) q[0];
rz(1.9908028929797368) q[0];
ry(0.3885864661272782) q[1];
rz(-1.9897290997613801) q[1];
ry(-0.03590066753534229) q[2];
rz(-0.9064847950504278) q[2];
ry(0.13829896289003768) q[3];
rz(1.4611607266524265) q[3];
ry(-1.3257333551101123) q[4];
rz(0.14821821130102822) q[4];
ry(-0.7203564088128465) q[5];
rz(3.125061363436741) q[5];
ry(-0.5471266822862901) q[6];
rz(-1.9034915967796955) q[6];
ry(-2.1924052329022667) q[7];
rz(-3.1121354908237437) q[7];
ry(0.852439232910486) q[8];
rz(-2.0451959476377315) q[8];
ry(3.0853079908281065) q[9];
rz(-2.2523286005411833) q[9];
ry(3.1253523330935598) q[10];
rz(-1.5523927636928216) q[10];
ry(-1.0490111207465551) q[11];
rz(0.4395400998292436) q[11];
ry(0.0019803179485249345) q[12];
rz(-1.0773873587757155) q[12];
ry(-1.246426607234456) q[13];
rz(-2.19598326274577) q[13];
ry(1.5403730179907384) q[14];
rz(-0.006669465929280392) q[14];
ry(1.0049494338535938) q[15];
rz(-0.02324026731987783) q[15];
ry(3.1234047670157916) q[16];
rz(-1.64832601213994) q[16];
ry(-0.8423113852014987) q[17];
rz(-1.5987489154386276) q[17];
ry(-0.11920594386736628) q[18];
rz(0.6947136328179564) q[18];
ry(-1.423364177605374) q[19];
rz(0.3284643927816975) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.805132222839962) q[0];
rz(-2.828905684525301) q[0];
ry(2.6171870778402555) q[1];
rz(0.3538208418127025) q[1];
ry(1.1926193319184577) q[2];
rz(-0.4986372990564565) q[2];
ry(1.5345542862046548) q[3];
rz(1.3846853658903762) q[3];
ry(0.21694264397418964) q[4];
rz(2.3807193012357115) q[4];
ry(2.867462570749106) q[5];
rz(1.2301800816413229) q[5];
ry(0.14934762681853808) q[6];
rz(-0.4130111695602864) q[6];
ry(-2.9454210421064895) q[7];
rz(2.513782637351646) q[7];
ry(-0.1518104348188416) q[8];
rz(-2.953732057884729) q[8];
ry(-3.115003015240639) q[9];
rz(0.8132764600414369) q[9];
ry(-1.0375760784432864) q[10];
rz(1.537200054240773) q[10];
ry(0.5879443957438626) q[11];
rz(-0.4892149179051514) q[11];
ry(-0.018516222071254737) q[12];
rz(1.0945846310333776) q[12];
ry(-1.5138927359543703) q[13];
rz(-0.14456227121462153) q[13];
ry(0.03460731828062827) q[14];
rz(-2.5460204229006207) q[14];
ry(-1.5760156583236418) q[15];
rz(-3.113135250773539) q[15];
ry(-0.18727974967630348) q[16];
rz(-0.32230271367155106) q[16];
ry(0.6899434975391081) q[17];
rz(-2.0045199483946634) q[17];
ry(2.972056995383186) q[18];
rz(-1.920991929622967) q[18];
ry(-0.6128682100191378) q[19];
rz(-0.11873959671557605) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.769709234098989) q[0];
rz(1.9327219781969913) q[0];
ry(3.140496322659666) q[1];
rz(0.3535303949250573) q[1];
ry(2.7653828419425626) q[2];
rz(2.923522544282137) q[2];
ry(-1.7778358398077607) q[3];
rz(0.07715753091043176) q[3];
ry(-1.4215754106935603) q[4];
rz(-1.2643852557541562) q[4];
ry(1.528171813406943) q[5];
rz(0.964242129591077) q[5];
ry(2.686711460702824) q[6];
rz(-1.1015356847807531) q[6];
ry(-2.152170444673072) q[7];
rz(2.884960464826143) q[7];
ry(-1.9663774404676682) q[8];
rz(1.478846282561734) q[8];
ry(2.8183041180469592) q[9];
rz(2.986014013879754) q[9];
ry(-0.22537752226528432) q[10];
rz(0.17414179083832693) q[10];
ry(2.8496004553520726) q[11];
rz(-0.5527544817507355) q[11];
ry(-0.025209561916492127) q[12];
rz(2.5290425406493577) q[12];
ry(0.575049088119552) q[13];
rz(1.6677681905241746) q[13];
ry(3.1391193666286177) q[14];
rz(-0.9838780898520303) q[14];
ry(2.8438277995394046) q[15];
rz(1.5412443431633163) q[15];
ry(1.5555464522153013) q[16];
rz(-1.0439653128546773) q[16];
ry(-1.216564042689261) q[17];
rz(3.127910698762894) q[17];
ry(-3.0573894006964255) q[18];
rz(-2.574535539385751) q[18];
ry(1.832172047140856) q[19];
rz(0.8560004378482519) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.025893561960368) q[0];
rz(-2.509141760067842) q[0];
ry(1.7147080048637626) q[1];
rz(-2.4550478002138205) q[1];
ry(-3.121481736974051) q[2];
rz(2.6245692626161685) q[2];
ry(-1.491413264935331) q[3];
rz(-2.9282257771792235) q[3];
ry(2.68227654419381) q[4];
rz(2.3421716271498343) q[4];
ry(2.5323384228466135) q[5];
rz(0.27906426987996613) q[5];
ry(-0.1684610340476551) q[6];
rz(-0.3495593084125428) q[6];
ry(2.855633105870525) q[7];
rz(-2.252519670755192) q[7];
ry(-3.0069934988618483) q[8];
rz(1.6804480996319127) q[8];
ry(-0.02512535064603666) q[9];
rz(-0.4140043634989299) q[9];
ry(2.967564925996505) q[10];
rz(1.4071913936791454) q[10];
ry(-0.8206944526684096) q[11];
rz(2.3160705377967585) q[11];
ry(-3.1356694934198375) q[12];
rz(-1.2730956865905998) q[12];
ry(1.090539386970489) q[13];
rz(-0.10183945231364078) q[13];
ry(-0.9625759440443984) q[14];
rz(-0.009839429750689632) q[14];
ry(-0.002482076800122251) q[15];
rz(-1.6823495585578245) q[15];
ry(0.0032788085454749593) q[16];
rz(0.060040503322125814) q[16];
ry(-1.5274585160272862) q[17];
rz(-1.9949660771179582e-05) q[17];
ry(1.5972969326279767) q[18];
rz(0.20306811260240923) q[18];
ry(-0.3478827618330884) q[19];
rz(-0.23704856511683367) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.6062512610520567) q[0];
rz(2.0615740014329096) q[0];
ry(-3.0033634637687743) q[1];
rz(1.8654488485014458) q[1];
ry(-1.6853037660796022) q[2];
rz(-1.6250879704585746) q[2];
ry(2.9591428127139308) q[3];
rz(2.025858618770242) q[3];
ry(0.0507060513255535) q[4];
rz(0.5647218219973302) q[4];
ry(0.29889483203124534) q[5];
rz(-0.3139482078568365) q[5];
ry(3.033539170936617) q[6];
rz(-2.9656463077665385) q[6];
ry(0.7440133924685863) q[7];
rz(-2.9815868226630946) q[7];
ry(-1.756076764619932) q[8];
rz(1.2354794910771265) q[8];
ry(-1.3909475113212573) q[9];
rz(-0.1585535087957502) q[9];
ry(0.8603829309995804) q[10];
rz(-1.9748237241520483) q[10];
ry(1.742750386500743) q[11];
rz(-0.5836919418855798) q[11];
ry(1.6046683394924621) q[12];
rz(-1.0413290307287273) q[12];
ry(-2.705500410559101) q[13];
rz(3.0148778723102665) q[13];
ry(-0.9307119793350989) q[14];
rz(3.0509828708421654) q[14];
ry(0.0339070293230854) q[15];
rz(1.7577702190839328) q[15];
ry(-0.010721542094701777) q[16];
rz(-2.1399612062649096) q[16];
ry(-1.7914843032290229) q[17];
rz(-0.49503856188742645) q[17];
ry(-3.1332623668565622) q[18];
rz(1.8239097557892447) q[18];
ry(-1.9840168151842796) q[19];
rz(-1.2363963097713608) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.4863728460665615) q[0];
rz(-0.2712265995303628) q[0];
ry(3.12615564461394) q[1];
rz(-2.189791898798165) q[1];
ry(0.3007215217303602) q[2];
rz(-0.10474858984397438) q[2];
ry(-0.4743021086484109) q[3];
rz(-0.11253454563961096) q[3];
ry(0.39466715427969873) q[4];
rz(2.4943731496055683) q[4];
ry(-0.4610840896034975) q[5];
rz(1.140100693250604) q[5];
ry(2.9278144264682657) q[6];
rz(1.1127250145084757) q[6];
ry(-0.2238762601542004) q[7];
rz(-1.036683545758736) q[7];
ry(-0.07267435090769768) q[8];
rz(-0.42306039253686933) q[8];
ry(-3.0762662312487556) q[9];
rz(-0.20546491582038495) q[9];
ry(0.2651682084116018) q[10];
rz(-1.6061207892217126) q[10];
ry(2.836652701573199) q[11];
rz(-1.5954106534360804) q[11];
ry(2.995247771142902) q[12];
rz(-2.8457633437424246) q[12];
ry(0.10268449348061273) q[13];
rz(-2.689336040764982) q[13];
ry(0.4088297472652424) q[14];
rz(-1.2561605491537553) q[14];
ry(0.7168643488200217) q[15];
rz(-3.0691581664427794) q[15];
ry(-2.9870509194092607) q[16];
rz(-1.5687381096579294) q[16];
ry(0.046005298289336416) q[17];
rz(2.6643131162818707) q[17];
ry(-0.01604713958289116) q[18];
rz(1.2146826491895981) q[18];
ry(-1.0977763432137537) q[19];
rz(1.8049533891822316) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.6540835974181258) q[0];
rz(1.293041690048735) q[0];
ry(-0.9843937005653087) q[1];
rz(-0.28314710178410607) q[1];
ry(0.3319191370961674) q[2];
rz(0.13225133098738162) q[2];
ry(3.123652360009701) q[3];
rz(0.32117634334249257) q[3];
ry(0.004564224768778935) q[4];
rz(-2.1162333572272214) q[4];
ry(-2.09249373635443) q[5];
rz(1.576753226391168) q[5];
ry(0.611812151707517) q[6];
rz(-1.516824297607113) q[6];
ry(3.1389096933885092) q[7];
rz(-1.0552627670930994) q[7];
ry(-2.7844488175722333) q[8];
rz(1.8300845913522759) q[8];
ry(-0.060607998160731764) q[9];
rz(3.0808060593855293) q[9];
ry(-0.9334173037662152) q[10];
rz(-1.0555290648410456) q[10];
ry(0.14140993725756595) q[11];
rz(2.5693798400918393) q[11];
ry(1.4374547600381655) q[12];
rz(-2.6899053288800374) q[12];
ry(-2.8787918101790044) q[13];
rz(-2.0166393121548882) q[13];
ry(-3.1056884025581906) q[14];
rz(-2.8354859561411736) q[14];
ry(1.2551812802459337) q[15];
rz(-0.002378998756447824) q[15];
ry(0.10271057202837941) q[16];
rz(-1.6088213762874917) q[16];
ry(-1.5600057332243091) q[17];
rz(-0.015222686816264748) q[17];
ry(1.5670059736814634) q[18];
rz(1.7924150815550286) q[18];
ry(-0.06708676954527881) q[19];
rz(0.4982790299462598) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.9084193704686824) q[0];
rz(-0.6639917864032344) q[0];
ry(-1.617836465521303) q[1];
rz(1.539544874689363) q[1];
ry(-0.10382157190615804) q[2];
rz(0.06811815098759144) q[2];
ry(0.5891975470234365) q[3];
rz(-1.4166328868070546) q[3];
ry(0.2282124297456587) q[4];
rz(-2.1796202794092854) q[4];
ry(-0.09485489753115355) q[5];
rz(0.7810793404835552) q[5];
ry(0.02187914933306967) q[6];
rz(2.547551796168844) q[6];
ry(2.9899448252815763) q[7];
rz(-1.9525632447115784) q[7];
ry(-0.005730042203006377) q[8];
rz(1.1132340693514182) q[8];
ry(-0.060804150025252746) q[9];
rz(-1.775259271155588) q[9];
ry(0.2317589037659265) q[10];
rz(2.5348814062043266) q[10];
ry(2.9563780938582944) q[11];
rz(-0.20129061711876164) q[11];
ry(3.017526148270533) q[12];
rz(3.1278847909337677) q[12];
ry(3.0528162257919655) q[13];
rz(-0.970137439389498) q[13];
ry(-0.09796884645681915) q[14];
rz(3.0792881504772183) q[14];
ry(-0.715901085423976) q[15];
rz(-0.23818676591875043) q[15];
ry(-3.1362326756861876) q[16];
rz(1.5167136429225179) q[16];
ry(0.002001020654867201) q[17];
rz(1.6095975629909947) q[17];
ry(0.005482523615631493) q[18];
rz(2.917319743454069) q[18];
ry(1.5757096312571621) q[19];
rz(-1.5711660469725524) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.1359338123578135) q[0];
rz(0.14898524813671887) q[0];
ry(-1.578067299913025) q[1];
rz(-1.2152484263819954) q[1];
ry(-1.5962936701790058) q[2];
rz(2.034595430773064) q[2];
ry(0.14584559303422392) q[3];
rz(-1.923279651848656) q[3];
ry(-3.014263034438247) q[4];
rz(-0.6156216115157395) q[4];
ry(-2.1723155897261153) q[5];
rz(-1.5971041411418323) q[5];
ry(-1.839285824733278) q[6];
rz(-0.9896985405289094) q[6];
ry(2.210979611971183) q[7];
rz(-0.3972314539425765) q[7];
ry(0.829973328719495) q[8];
rz(1.6385175330197184) q[8];
ry(1.919895340158295) q[9];
rz(-1.2482931929909562) q[9];
ry(2.298877137621334) q[10];
rz(1.5366342232822747) q[10];
ry(-1.03520442604544) q[11];
rz(0.9031836652435086) q[11];
ry(0.6575433260914219) q[12];
rz(1.2406259040120022) q[12];
ry(-1.565306574343362) q[13];
rz(2.59576427602217) q[13];
ry(1.5847444291550428) q[14];
rz(2.633299225567065) q[14];
ry(-0.3133852924644493) q[15];
rz(2.810985355881678) q[15];
ry(-1.50896627824444) q[16];
rz(-2.0650643827454944) q[16];
ry(2.540656002341178) q[17];
rz(2.598990039826112) q[17];
ry(1.5661357498219524) q[18];
rz(-2.397929368289235) q[18];
ry(-1.5757688516075876) q[19];
rz(-1.8780340739554278) q[19];