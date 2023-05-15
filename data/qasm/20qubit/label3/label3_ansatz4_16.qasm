OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.1332895613387364) q[0];
rz(-1.4188427924918867) q[0];
ry(1.3379580868867003) q[1];
rz(2.9690920272887267) q[1];
ry(0.7897276674706096) q[2];
rz(-3.08679233122645) q[2];
ry(1.4902287299645238) q[3];
rz(2.4921513511058397) q[3];
ry(1.4761008959030313) q[4];
rz(-0.011172190272891848) q[4];
ry(1.4669666633971543) q[5];
rz(-0.13784227866132515) q[5];
ry(-3.141585940256039) q[6];
rz(-2.5065927975316176) q[6];
ry(-0.31643307612038285) q[7];
rz(-1.8570137499150097) q[7];
ry(1.5703696251091468) q[8];
rz(-2.0239772438051737) q[8];
ry(-1.5720927820441672) q[9];
rz(-3.1404845245775586) q[9];
ry(-0.0003122116363477129) q[10];
rz(-3.135719922064963) q[10];
ry(0.0008655029638873922) q[11];
rz(-1.243446536849352) q[11];
ry(1.2580998890354136) q[12];
rz(-2.2162965674793798) q[12];
ry(-1.301233017812157) q[13];
rz(3.124451050464946) q[13];
ry(-3.141314670363171) q[14];
rz(-0.320929972085112) q[14];
ry(0.00480388517339847) q[15];
rz(-0.9043605130688261) q[15];
ry(-1.9879550604611007) q[16];
rz(1.494350959106158) q[16];
ry(-1.3751894917500496) q[17];
rz(1.4370367059487585) q[17];
ry(-3.0092017371979676) q[18];
rz(2.559716223581902) q[18];
ry(1.5155963715903635) q[19];
rz(-3.0007207666330005) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.880059759208291) q[0];
rz(-2.557676019165953) q[0];
ry(1.2997800025777875) q[1];
rz(-2.381816288538009) q[1];
ry(-2.483748535300905) q[2];
rz(-2.0305991324145554) q[2];
ry(0.22047151336764528) q[3];
rz(1.373097308570939) q[3];
ry(-0.5687688871255965) q[4];
rz(1.731391600057444) q[4];
ry(0.6001714812464041) q[5];
rz(-3.0802817150939803) q[5];
ry(1.5662500205178294) q[6];
rz(-3.1210501967193855) q[6];
ry(0.0007583469594792268) q[7];
rz(-2.3599398158360296) q[7];
ry(-0.0011398590465212521) q[8];
rz(2.0255144364197513) q[8];
ry(-1.5708307992869817) q[9];
rz(0.0005936271847492603) q[9];
ry(-9.227687629257905e-05) q[10];
rz(-1.3204190467821793) q[10];
ry(-3.1409238699393742) q[11];
rz(-2.6634192154476097) q[11];
ry(-0.435600506719175) q[12];
rz(-0.4132512473309591) q[12];
ry(2.4278792203646007) q[13];
rz(3.102358006599771) q[13];
ry(0.0011810611949671923) q[14];
rz(0.16151381441417373) q[14];
ry(-2.7210037174749555) q[15];
rz(-1.424825989538362) q[15];
ry(-2.1164579801058805) q[16];
rz(2.863577182737859) q[16];
ry(-2.7028345106066456) q[17];
rz(0.9949799020149254) q[17];
ry(2.3963081970582643) q[18];
rz(-0.1504312728160624) q[18];
ry(1.25616708521272) q[19];
rz(-2.215499890355489) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.9586957166755847) q[0];
rz(2.828621310706124) q[0];
ry(2.680143178750808) q[1];
rz(0.19327083590355595) q[1];
ry(2.5672887413854815) q[2];
rz(-2.589891933898648) q[2];
ry(2.4576435906052843) q[3];
rz(-1.3086076164545093) q[3];
ry(0.02184390458755381) q[4];
rz(-0.9427556759714708) q[4];
ry(0.07684938153710963) q[5];
rz(3.021717505431709) q[5];
ry(1.5174191690951124) q[6];
rz(-1.932272068516803) q[6];
ry(-3.131749343803251) q[7];
rz(-1.0293764687436493) q[7];
ry(1.564991338491275) q[8];
rz(-0.01958915710959719) q[8];
ry(-1.5668380091861698) q[9];
rz(-2.665212661446533) q[9];
ry(-3.1398563704276055) q[10];
rz(2.6781016338585877) q[10];
ry(-0.0012381575546031383) q[11];
rz(-2.0173140496612314) q[11];
ry(2.1692906988454546) q[12];
rz(-2.186734241046846) q[12];
ry(0.8144244256427151) q[13];
rz(-0.4749884489051958) q[13];
ry(3.1415264529374856) q[14];
rz(-0.5126039958224027) q[14];
ry(0.0005782785346562335) q[15];
rz(0.4800702466132556) q[15];
ry(1.7532624088986295) q[16];
rz(1.521953576120602) q[16];
ry(1.1655520223697087) q[17];
rz(-0.7554973918103823) q[17];
ry(0.8951034844838006) q[18];
rz(2.504414500781318) q[18];
ry(0.3183763442830738) q[19];
rz(2.648088450361443) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.8038060877985735) q[0];
rz(2.310200056028178) q[0];
ry(2.04040499356243) q[1];
rz(3.131838107396681) q[1];
ry(-2.670665521781741) q[2];
rz(-0.8133078454150213) q[2];
ry(-0.44693971902252905) q[3];
rz(1.2923918692336118) q[3];
ry(3.120734469646724) q[4];
rz(1.1889710195742706) q[4];
ry(0.29444029598685884) q[5];
rz(-2.872824059940258) q[5];
ry(-1.5071419313764411) q[6];
rz(-2.9562709380129766) q[6];
ry(-1.5650878588275523) q[7];
rz(0.41050446138199614) q[7];
ry(-0.003554329730220296) q[8];
rz(-0.902422440998651) q[8];
ry(-0.005738122508385497) q[9];
rz(-2.5825005128976017) q[9];
ry(-1.6664082954208919) q[10];
rz(0.058506553577927896) q[10];
ry(3.0994271514011027) q[11];
rz(-2.9517535357324567) q[11];
ry(0.5560032222597098) q[12];
rz(-1.6923658663597214) q[12];
ry(-2.2659489950905454) q[13];
rz(2.340047385334913) q[13];
ry(-3.1374554857442893) q[14];
rz(-2.174520126894242) q[14];
ry(-1.5665454031343709) q[15];
rz(1.6890301247993456) q[15];
ry(-2.538218061404113) q[16];
rz(2.595697074085396) q[16];
ry(1.2037679262477894) q[17];
rz(-0.820685848724209) q[17];
ry(-1.1029657891105933) q[18];
rz(0.35948733208100053) q[18];
ry(0.18868191288505154) q[19];
rz(-2.1631320134186525) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.7268560492926883) q[0];
rz(2.172609950944496) q[0];
ry(-1.7561085877464269) q[1];
rz(1.226700089615743) q[1];
ry(-2.122277071887602) q[2];
rz(-2.3168097755434722) q[2];
ry(-1.3948525623006116) q[3];
rz(3.106514769326108) q[3];
ry(-1.5722685922600075) q[4];
rz(-3.141001193973068) q[4];
ry(3.140905580217398) q[5];
rz(0.2996734795678497) q[5];
ry(-1.5706207085226283) q[6];
rz(1.573314660420432) q[6];
ry(-0.0011311119062762387) q[7];
rz(1.1864763062114194) q[7];
ry(-3.1394758588286384) q[8];
rz(-0.2212681811260729) q[8];
ry(3.1178339134217685) q[9];
rz(1.740135238888079) q[9];
ry(1.6200195932345365) q[10];
rz(2.392895214805649) q[10];
ry(-0.17586887223230452) q[11];
rz(-2.547167102788959) q[11];
ry(6.451108587267316e-05) q[12];
rz(0.5838094471193804) q[12];
ry(-0.0008998229140848579) q[13];
rz(2.687997202631649) q[13];
ry(0.0013107357842361955) q[14];
rz(1.9069355175119742) q[14];
ry(-2.930595518929276) q[15];
rz(-2.3900936210501267) q[15];
ry(-1.360950463724749) q[16];
rz(1.6080697207131465) q[16];
ry(-2.411677462779295) q[17];
rz(0.2934491426199797) q[17];
ry(-2.4971120988152005) q[18];
rz(0.9921678839577767) q[18];
ry(2.3240064850548516) q[19];
rz(0.6449642242047634) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.0503728084199588) q[0];
rz(-0.1909309243846833) q[0];
ry(2.2875159019026867) q[1];
rz(-1.6282730773761225) q[1];
ry(-0.00017152385018635505) q[2];
rz(-2.410330381456404) q[2];
ry(-0.0001726094512475794) q[3];
rz(2.7323906856898557) q[3];
ry(1.5717760099212361) q[4];
rz(0.5851262132989969) q[4];
ry(2.513373382495005) q[5];
rz(-0.3191414679234578) q[5];
ry(1.390144358185206) q[6];
rz(3.1406658564880785) q[6];
ry(0.054110368587728885) q[7];
rz(1.5653738604839154) q[7];
ry(-3.1289915482705197) q[8];
rz(-0.01591115252232811) q[8];
ry(-0.014603383389060198) q[9];
rz(-1.7888196524572333) q[9];
ry(-2.0805900652928706) q[10];
rz(1.4894041211167766) q[10];
ry(-1.4573306700174964) q[11];
rz(-1.9802811604569726) q[11];
ry(-0.0023995369661591326) q[12];
rz(2.953226498324512) q[12];
ry(3.140355907131758) q[13];
rz(-3.0265362902210344) q[13];
ry(3.1415179731803957) q[14];
rz(0.9039138317250324) q[14];
ry(0.5237036011379623) q[15];
rz(-0.058713122673549556) q[15];
ry(-2.633201912771917) q[16];
rz(-2.558786247410947) q[16];
ry(-0.24900606862614705) q[17];
rz(2.882896772202114) q[17];
ry(-2.1159488693354827) q[18];
rz(-0.062247750019380106) q[18];
ry(2.301123687702881) q[19];
rz(-0.8560759301328008) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.4581869948110016) q[0];
rz(0.8115989703041248) q[0];
ry(2.440310543068358) q[1];
rz(2.9338116179370415) q[1];
ry(-0.0009503940592034965) q[2];
rz(1.0100735874720725) q[2];
ry(5.1400835988424376e-05) q[3];
rz(-0.07510875839905976) q[3];
ry(5.025731193608607e-05) q[4];
rz(-2.5971022549535894) q[4];
ry(3.089298203863981) q[5];
rz(2.69251764558444) q[5];
ry(-1.5732560326839067) q[6];
rz(-1.4768349429846026) q[6];
ry(-3.0385193881159216) q[7];
rz(0.12498126579159675) q[7];
ry(3.1098222281903327) q[8];
rz(1.1272664520474747) q[8];
ry(-1.6005867767411281) q[9];
rz(-2.661624969291652) q[9];
ry(1.4880934770093805) q[10];
rz(1.1230702482894415) q[10];
ry(1.645093067567049) q[11];
rz(-1.617629625875849) q[11];
ry(-3.141098381679076) q[12];
rz(2.1975158499420537) q[12];
ry(-3.140446815114871) q[13];
rz(1.5354395200076323) q[13];
ry(-3.1397284037439492) q[14];
rz(-2.5342705735567512) q[14];
ry(-0.22470307800629818) q[15];
rz(-0.5337126988849575) q[15];
ry(-0.9397850767845969) q[16];
rz(-2.4583110444339336) q[16];
ry(2.68755506516659) q[17];
rz(-2.306196155429993) q[17];
ry(1.7428919449808191) q[18];
rz(3.087129188410691) q[18];
ry(-2.070691744969789) q[19];
rz(0.16666963409950236) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.5643116970931965) q[0];
rz(-2.8602027176834106) q[0];
ry(-2.193460654295756) q[1];
rz(1.7926963929028141) q[1];
ry(-0.0010376558878597566) q[2];
rz(1.965500937329186) q[2];
ry(3.140936103421593) q[3];
rz(2.508877135806659) q[3];
ry(1.3952744970397797) q[4];
rz(-1.6103903297561208) q[4];
ry(1.5695222840020784) q[5];
rz(1.571524942601231) q[5];
ry(-0.00027917645864295956) q[6];
rz(1.738966072970126) q[6];
ry(-3.141324441679323) q[7];
rz(-1.0288231173860447) q[7];
ry(-0.0030734049475276137) q[8];
rz(-1.3634069270881275) q[8];
ry(0.005552090065782167) q[9];
rz(-0.4700216402552731) q[9];
ry(3.12041619666394) q[10];
rz(0.9385205003239339) q[10];
ry(1.6606011076137097) q[11];
rz(0.03478242139404131) q[11];
ry(2.550635156430775) q[12];
rz(-2.4224352647320027) q[12];
ry(3.140740774673237) q[13];
rz(2.4478334729673494) q[13];
ry(3.141245397068478) q[14];
rz(-1.7425709720913414) q[14];
ry(-2.7154147559424215) q[15];
rz(-1.3302781173685627) q[15];
ry(2.5399498102014753) q[16];
rz(0.9012059745036538) q[16];
ry(2.54630147890314) q[17];
rz(-1.1721362849471748) q[17];
ry(1.6032175151518675) q[18];
rz(2.089375242050372) q[18];
ry(-1.6028152856239128) q[19];
rz(-2.8892060399920623) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.7492477303699787) q[0];
rz(1.864675231409227) q[0];
ry(-0.6241112068627661) q[1];
rz(1.8902213147204776) q[1];
ry(-1.6117409083126462) q[2];
rz(2.2262075887057033) q[2];
ry(1.6489966968603478) q[3];
rz(1.8762925502897785) q[3];
ry(-1.8193672109811525) q[4];
rz(2.2864699306691376) q[4];
ry(1.5726575913208007) q[5];
rz(1.6750828581147938) q[5];
ry(-1.9380979135429053) q[6];
rz(-3.0292153226597818) q[6];
ry(0.003594041528839083) q[7];
rz(-1.8168186422957795) q[7];
ry(1.5659659472020167) q[8];
rz(2.1743947163910806) q[8];
ry(0.47252143875131364) q[9];
rz(-1.7346796800002586) q[9];
ry(3.136580352810561) q[10];
rz(1.8841558901225095) q[10];
ry(0.4687899720163293) q[11];
rz(-1.6171093190139807) q[11];
ry(-0.00019486397056223126) q[12];
rz(2.434679937796987) q[12];
ry(7.679756710916277e-06) q[13];
rz(1.1692957248442568) q[13];
ry(0.006860227459975229) q[14];
rz(-3.0340517735100923) q[14];
ry(-3.136560287673616) q[15];
rz(2.037388322270946) q[15];
ry(-0.5310016799106794) q[16];
rz(0.5750561038003487) q[16];
ry(2.736894801744623) q[17];
rz(-0.7890601295328242) q[17];
ry(2.5011014621085277) q[18];
rz(-0.03368450587431493) q[18];
ry(-0.4975412085471538) q[19];
rz(1.4999370478683325) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.0027724903996267976) q[0];
rz(-2.0774761044797194) q[0];
ry(-2.8734574066560814) q[1];
rz(0.08399730492786384) q[1];
ry(-2.3056092192596) q[2];
rz(1.94046235368844) q[2];
ry(-0.769748004147762) q[3];
rz(0.8632028103860389) q[3];
ry(3.14050301408448) q[4];
rz(-2.8478948166249296) q[4];
ry(2.5951757855542787) q[5];
rz(-1.6795687110257294) q[5];
ry(2.9364802226778406) q[6];
rz(-0.11803651535787994) q[6];
ry(0.0017501239573470965) q[7];
rz(0.5675152838464823) q[7];
ry(0.0008087494662936765) q[8];
rz(-0.5635254854478147) q[8];
ry(-0.0015249350332888767) q[9];
rz(-0.06397872089735764) q[9];
ry(-0.004350061852635534) q[10];
rz(1.0059426091275598) q[10];
ry(1.5645184643653236) q[11];
rz(3.125593724972677) q[11];
ry(2.117970188634441) q[12];
rz(-1.4241378874062307) q[12];
ry(3.1398961621554844) q[13];
rz(-2.324179625722998) q[13];
ry(0.00043193771476843744) q[14];
rz(-2.380168170171447) q[14];
ry(-1.3419182755966688) q[15];
rz(-1.3443670612488807) q[15];
ry(-1.1178686234375084) q[16];
rz(3.06906776663089) q[16];
ry(-2.4394872997168715) q[17];
rz(-2.5303600341993953) q[17];
ry(1.6596299623514061) q[18];
rz(-1.8083560380938462) q[18];
ry(0.7942638202576503) q[19];
rz(2.292767085902092) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.008310264454024) q[0];
rz(-2.0679926914818214) q[0];
ry(1.6370159696754558) q[1];
rz(1.5867563311266508) q[1];
ry(0.9229792320935708) q[2];
rz(0.6248804632355788) q[2];
ry(1.3645917114985062) q[3];
rz(0.5164869979349779) q[3];
ry(0.08000717424637038) q[4];
rz(0.4243691783351689) q[4];
ry(-0.0007414160652920998) q[5];
rz(-1.5780233463916815) q[5];
ry(1.2711919221083945) q[6];
rz(-2.625756218092422) q[6];
ry(-0.00532415719198287) q[7];
rz(-2.345472409270252) q[7];
ry(-1.5727395170414358) q[8];
rz(0.3535257825369813) q[8];
ry(-3.10732804052964) q[9];
rz(-0.9600668202945499) q[9];
ry(3.1386630616148854) q[10];
rz(1.233544914355229) q[10];
ry(-0.0488311191650137) q[11];
rz(-1.6048418064622105) q[11];
ry(0.21626669817313068) q[12];
rz(3.0002003257701992) q[12];
ry(3.1366356176803496) q[13];
rz(-0.6118319804296749) q[13];
ry(1.6694746582712705) q[14];
rz(-1.786985047221432) q[14];
ry(0.0031013422815213687) q[15];
rz(-2.777245401053577) q[15];
ry(-0.5156471643928199) q[16];
rz(-1.050381011862197) q[16];
ry(2.82366425040146) q[17];
rz(2.5936234751718485) q[17];
ry(1.932731565370205) q[18];
rz(-2.6100776869952065) q[18];
ry(-0.7159816452916739) q[19];
rz(-0.12141681528143877) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.821515830249404) q[0];
rz(-0.7713918780022684) q[0];
ry(0.9521826583988302) q[1];
rz(-0.38581338952947863) q[1];
ry(1.8500996066204491) q[2];
rz(1.9666430867479034) q[2];
ry(-1.138374454047275) q[3];
rz(-0.31097691678003514) q[3];
ry(-0.00017608053054551928) q[4];
rz(1.1516813499183127) q[4];
ry(3.0406100391886532) q[5];
rz(0.0865003169672347) q[5];
ry(-3.137001470323998) q[6];
rz(1.1323812549673895) q[6];
ry(1.5435431400487374) q[7];
rz(-1.1580645967837209) q[7];
ry(-3.1396773255823707) q[8];
rz(-2.5846674383844257) q[8];
ry(-0.0016177392375574609) q[9];
rz(0.49668579003456365) q[9];
ry(1.557309349246502) q[10];
rz(-0.40667604144919656) q[10];
ry(1.6236847843218172) q[11];
rz(-2.42842847746139) q[11];
ry(2.6296887750452926) q[12];
rz(1.8265714435744416) q[12];
ry(1.5950175989680515) q[13];
rz(-2.3236991446303166) q[13];
ry(0.0006703516522525987) q[14];
rz(-1.4939429631190935) q[14];
ry(3.1330321001987493) q[15];
rz(-0.40359416380528457) q[15];
ry(0.009635319212701554) q[16];
rz(2.937502572185533) q[16];
ry(1.563427083823905) q[17];
rz(1.569462358764456) q[17];
ry(-1.4475263758046752) q[18];
rz(0.33676064537787503) q[18];
ry(-2.9851076863083246) q[19];
rz(1.5690113940488262) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5424402310125647) q[0];
rz(0.10445098018048869) q[0];
ry(2.2014940112981165) q[1];
rz(1.7969286134970446) q[1];
ry(-2.0304690101444307) q[2];
rz(-1.806837286688681) q[2];
ry(-1.4583445094717469) q[3];
rz(0.8497190304456765) q[3];
ry(-0.00041990475535491595) q[4];
rz(0.18234219856323008) q[4];
ry(-0.007746172411945739) q[5];
rz(-0.5014507560397268) q[5];
ry(0.0009132694238678242) q[6];
rz(-2.1974373302982824) q[6];
ry(3.1304031910991585) q[7];
rz(1.9615254594643103) q[7];
ry(2.921909530887558) q[8];
rz(-0.3142022810450582) q[8];
ry(0.01365226834715294) q[9];
rz(2.1492811708387505) q[9];
ry(-3.141323047447319) q[10];
rz(2.7279168519511434) q[10];
ry(-3.1415689121930943) q[11];
rz(-0.8513345863291527) q[11];
ry(-0.0007126553339227447) q[12];
rz(1.3165611478409007) q[12];
ry(-3.140272533668158) q[13];
rz(0.9682775050374023) q[13];
ry(-0.0008235487310699341) q[14];
rz(-0.8199951010132596) q[14];
ry(0.14122550075522824) q[15];
rz(-1.3159465243634747) q[15];
ry(1.5333360151690814) q[16];
rz(-3.048964535028807) q[16];
ry(1.5346115660452766) q[17];
rz(2.6463549878998007) q[17];
ry(1.5686424885330132) q[18];
rz(2.8975686350402254) q[18];
ry(-1.7821443574806501) q[19];
rz(2.7282358217956246) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.7570017009111778) q[0];
rz(3.0851475191491238) q[0];
ry(2.204074520763639) q[1];
rz(2.086590433656456) q[1];
ry(-0.9538144215803044) q[2];
rz(-0.9860072249560421) q[2];
ry(-1.7812966781868673) q[3];
rz(2.1259720615447826) q[3];
ry(-0.00020475788752438717) q[4];
rz(1.875871570612479) q[4];
ry(2.8615212833792505) q[5];
rz(-1.1141760578269224) q[5];
ry(-2.9419874726826536) q[6];
rz(-0.9445530115157028) q[6];
ry(-1.5416330262496405) q[7];
rz(0.9547208607326646) q[7];
ry(-0.002958787496091908) q[8];
rz(-2.2913208434349954) q[8];
ry(3.1375831944628594) q[9];
rz(-2.755343217761781) q[9];
ry(1.5760015040352346) q[10];
rz(-3.077981040009978) q[10];
ry(-1.4436985677059937) q[11];
rz(-2.5559786591144342) q[11];
ry(0.5097945067846895) q[12];
rz(-0.7898708405172614) q[12];
ry(-3.1179034512247488) q[13];
rz(-0.6193199926081293) q[13];
ry(-1.5695842954759962) q[14];
rz(-3.1396956727391157) q[14];
ry(1.4886561435571402) q[15];
rz(-3.132431387489254) q[15];
ry(3.090364442449993) q[16];
rz(0.9261729874715359) q[16];
ry(-3.1335446279784387) q[17];
rz(0.5933985082974473) q[17];
ry(0.09256441192051677) q[18];
rz(1.8121290617872399) q[18];
ry(-1.5964557590183446) q[19];
rz(0.23987782767359264) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.6326068950335311) q[0];
rz(-1.179204911911877) q[0];
ry(1.4546687023312523) q[1];
rz(-1.0601850084629383) q[1];
ry(1.6525136133228633) q[2];
rz(-3.0520638836452196) q[2];
ry(1.4220942234901068) q[3];
rz(1.1851621243948793) q[3];
ry(3.1275355011832024) q[4];
rz(-1.8547973339619253) q[4];
ry(3.1415335758096665) q[5];
rz(-0.6173631029245792) q[5];
ry(0.03415483024381772) q[6];
rz(1.0778997961621974) q[6];
ry(0.007790898859884984) q[7];
rz(-2.5307645718966665) q[7];
ry(-3.141278954438747) q[8];
rz(0.07534203321124505) q[8];
ry(-1.4933078324248585) q[9];
rz(-0.6789404125372563) q[9];
ry(-3.4667250970884115e-05) q[10];
rz(0.1775314766128231) q[10];
ry(0.0019926466422610445) q[11];
rz(0.5678956746904537) q[11];
ry(3.1413058336846356) q[12];
rz(2.294860721623715) q[12];
ry(0.003124131282866671) q[13];
rz(-2.3657746766915113) q[13];
ry(1.5724833558182238) q[14];
rz(-3.061931936119053) q[14];
ry(1.570812335198653) q[15];
rz(1.56915786375827) q[15];
ry(3.1410944701553243) q[16];
rz(-2.3256014284562276) q[16];
ry(-0.00011263248165540546) q[17];
rz(-1.4338074404679715) q[17];
ry(-1.5777448067291626) q[18];
rz(2.8125922455331462) q[18];
ry(-1.7820121129866635) q[19];
rz(2.9332081630748927) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.9222741265826732) q[0];
rz(1.5967835424441486) q[0];
ry(1.2975135460383254) q[1];
rz(1.4880246666476156) q[1];
ry(1.6735476756074374) q[2];
rz(0.5561563321436643) q[2];
ry(-2.4054884640206717) q[3];
rz(-1.476545344337087) q[3];
ry(-0.002755914207900112) q[4];
rz(0.1391912406886406) q[4];
ry(-1.5630308198777128) q[5];
rz(-0.9169737848864443) q[5];
ry(-3.114951360267945) q[6];
rz(1.0896054553556511) q[6];
ry(0.33330304059018) q[7];
rz(2.1202124509487383) q[7];
ry(0.003965516466520071) q[8];
rz(0.27098423984550135) q[8];
ry(0.005316475796392162) q[9];
rz(1.0951476090149819) q[9];
ry(1.5743098857427436) q[10];
rz(-3.0653724180262247) q[10];
ry(1.5760589653847124) q[11];
rz(2.5368945647526595) q[11];
ry(-0.6700036004964929) q[12];
rz(0.28875293265039925) q[12];
ry(-1.6272825102575679) q[13];
rz(1.3327534915404893) q[13];
ry(-1.5095101136468054) q[14];
rz(-1.3315101010985977) q[14];
ry(-0.9542110298334608) q[15];
rz(-1.5394206484915063) q[15];
ry(1.6053415691738948) q[16];
rz(3.0359160409966046) q[16];
ry(1.305956798281442) q[17];
rz(-1.4729983158029119) q[17];
ry(-1.7042002142362012) q[18];
rz(-1.2083262057107353) q[18];
ry(0.30124053137481965) q[19];
rz(0.26944879498216695) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.6718468944961609) q[0];
rz(-2.7306315969133697) q[0];
ry(-2.5206307325218336) q[1];
rz(2.951679850815871) q[1];
ry(0.11449013588722501) q[2];
rz(1.7317195247875077) q[2];
ry(-3.1255670156924467) q[3];
rz(1.1087381768437066) q[3];
ry(3.137531952261014) q[4];
rz(-2.7726458799209537) q[4];
ry(-3.1392681347524722) q[5];
rz(2.108950809139932) q[5];
ry(-3.075517368085477) q[6];
rz(-2.97516583410826) q[6];
ry(0.0019108950670070869) q[7];
rz(-0.14686515405258543) q[7];
ry(1.5632662594238744) q[8];
rz(-1.5391519986580793) q[8];
ry(3.1072246611636536) q[9];
rz(0.4857889673559193) q[9];
ry(-3.1414760497551644) q[10];
rz(-2.912714292599915) q[10];
ry(3.1415776191374203) q[11];
rz(0.9755573700462471) q[11];
ry(-6.996050278857278e-05) q[12];
rz(1.0929460969147633) q[12];
ry(-3.141592208520156) q[13];
rz(3.0736265676542596) q[13];
ry(0.002358416593476303) q[14];
rz(2.5125743317002613) q[14];
ry(3.103182950556886) q[15];
rz(-2.3463840522801864) q[15];
ry(-1.5708522340536595) q[16];
rz(-1.5706028920808208) q[16];
ry(-1.5699901412782848) q[17];
rz(1.5640397175747491) q[17];
ry(-1.5819879113967659) q[18];
rz(2.1430351804124896) q[18];
ry(1.5662376222025425) q[19];
rz(-0.4365491415263374) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.5812560837427051) q[0];
rz(-3.0595304949838966) q[0];
ry(1.8608677376742948) q[1];
rz(2.7632132351966257) q[1];
ry(-1.9240981404866055) q[2];
rz(0.10035909717566505) q[2];
ry(0.3491539486712734) q[3];
rz(2.12245057641845) q[3];
ry(-1.5677328051662267) q[4];
rz(-2.590823739223322) q[4];
ry(1.4972672179173312) q[5];
rz(0.7922624602317807) q[5];
ry(3.1413981156130637) q[6];
rz(0.2762802449007573) q[6];
ry(0.008589829066196764) q[7];
rz(-2.3113126821399206) q[7];
ry(1.57269491688139) q[8];
rz(-3.1393660620138366) q[8];
ry(1.5674554374635479) q[9];
rz(0.001231700761461063) q[9];
ry(-3.09959764549479) q[10];
rz(2.938270896754717) q[10];
ry(1.5696172957910157) q[11];
rz(-2.7290752217595466) q[11];
ry(-1.6888283419316048) q[12];
rz(1.2778049041875486) q[12];
ry(1.5388231054643162) q[13];
rz(2.1358028319484097) q[13];
ry(-0.009258867729005082) q[14];
rz(2.268374122018928) q[14];
ry(0.004337303777957935) q[15];
rz(1.3286382137677117) q[15];
ry(-1.8600226415651138) q[16];
rz(-1.5712954065945464) q[16];
ry(3.121956998132334) q[17];
rz(-0.046036420607402186) q[17];
ry(0.7245061612637231) q[18];
rz(-1.2873946421116953) q[18];
ry(-0.931188233054499) q[19];
rz(-2.2995830162823268) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.3084005998410584) q[0];
rz(-1.6279374306300238) q[0];
ry(1.6969329434446627) q[1];
rz(1.5315545865859281) q[1];
ry(-1.5700244048117211) q[2];
rz(1.564989235673884) q[2];
ry(-0.00012201163336071706) q[3];
rz(-3.110760002769796) q[3];
ry(3.1292900258444005) q[4];
rz(-1.0268898119562264) q[4];
ry(3.140579758531002) q[5];
rz(0.03644128894683482) q[5];
ry(0.00017031805134504197) q[6];
rz(-0.907314530147974) q[6];
ry(-3.1405907067245544) q[7];
rz(2.82545388309685) q[7];
ry(1.569918632612164) q[8];
rz(-1.5239324720966827) q[8];
ry(1.5713479855374026) q[9];
rz(1.101753715081978) q[9];
ry(-3.138480588716564) q[10];
rz(-1.925401077685505) q[10];
ry(-3.140709086567075) q[11];
rz(1.7566946618585162) q[11];
ry(3.1403655878713135) q[12];
rz(1.983055746153593) q[12];
ry(-3.1415491189628266) q[13];
rz(-0.8675776706696245) q[13];
ry(0.020064657482479653) q[14];
rz(1.353408268662431) q[14];
ry(3.1395668206700376) q[15];
rz(-0.9196795341348933) q[15];
ry(-1.5711986001225808) q[16];
rz(-1.502096549524123) q[16];
ry(0.10133624366980366) q[17];
rz(1.4603815817790882) q[17];
ry(3.11408348131654) q[18];
rz(-2.1991315516494425) q[18];
ry(-1.5836775773656018) q[19];
rz(-0.17228117352811825) q[19];
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
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.135992483679958) q[0];
rz(-1.4200998033242676) q[0];
ry(0.005573559139661732) q[1];
rz(1.6866306572783731) q[1];
ry(1.5712327601948415) q[2];
rz(0.13493484304694894) q[2];
ry(0.005720484041004781) q[3];
rz(1.8593407444153573) q[3];
ry(1.572163176384505) q[4];
rz(0.1670098649380074) q[4];
ry(0.008557189595794722) q[5];
rz(2.0744317203860154) q[5];
ry(3.1403414390868507) q[6];
rz(2.911687123440557) q[6];
ry(-1.5716855095830082) q[7];
rz(0.11251554743838015) q[7];
ry(0.08525899861308961) q[8];
rz(-0.4248020590693887) q[8];
ry(-3.015284492500602) q[9];
rz(-3.130957417551171) q[9];
ry(1.5578260794049357) q[10];
rz(2.7812461437510962) q[10];
ry(3.047518237196868) q[11];
rz(-0.10496109822033602) q[11];
ry(-1.346701729674309) q[12];
rz(-1.6654527839620576) q[12];
ry(-0.02699377995025473) q[13];
rz(-1.8660920091543032) q[13];
ry(1.562367185224039) q[14];
rz(1.3389529021057756) q[14];
ry(2.9849832412910695) q[15];
rz(-0.08660983499451785) q[15];
ry(1.5804543120895707) q[16];
rz(-1.799274847232792) q[16];
ry(-0.16476087605369746) q[17];
rz(-0.06614837373533877) q[17];
ry(1.5801653424245252) q[18];
rz(-1.796853028126252) q[18];
ry(1.5462133061325694) q[19];
rz(1.35663243203153) q[19];