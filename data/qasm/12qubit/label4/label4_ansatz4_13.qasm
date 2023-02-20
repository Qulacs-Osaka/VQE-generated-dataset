OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.512792885748792) q[0];
rz(0.04337213726083445) q[0];
ry(-0.005373404124368264) q[1];
rz(2.3852016009143653) q[1];
ry(0.54678964521431) q[2];
rz(2.571566670941908) q[2];
ry(3.1379817737381135) q[3];
rz(-1.128796291667997) q[3];
ry(1.5706487863974095) q[4];
rz(1.5707877410650821) q[4];
ry(-1.5672389096790882) q[5];
rz(1.5717241187069613) q[5];
ry(-3.1386143616913342) q[6];
rz(-2.532282375883699) q[6];
ry(-2.640474244597138) q[7];
rz(-1.1437474969632822) q[7];
ry(-0.3608509115185922) q[8];
rz(1.7002759808922299) q[8];
ry(3.137021326965199) q[9];
rz(0.5839937082922879) q[9];
ry(3.129365207169943) q[10];
rz(2.328987926012875) q[10];
ry(-1.5863296982380009) q[11];
rz(-1.5810215297916432) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.606831993644103) q[0];
rz(0.29843378084442906) q[0];
ry(-1.3389915492429187) q[1];
rz(0.44670282264680683) q[1];
ry(3.1376668069621543) q[2];
rz(1.3060633360098222) q[2];
ry(-0.0020480653394557144) q[3];
rz(2.97804869430016) q[3];
ry(-1.5703208454250548) q[4];
rz(-0.10863002004420591) q[4];
ry(1.5711430435121034) q[5];
rz(-1.8943140749295408) q[5];
ry(-1.570954867668504) q[6];
rz(3.140421257536373) q[6];
ry(-3.136425867602473) q[7];
rz(-2.1027883617905148) q[7];
ry(-0.011373349618238717) q[8];
rz(1.8621608379058066) q[8];
ry(0.34932807671982236) q[9];
rz(-2.151103718737433) q[9];
ry(-1.5706709814703408) q[10];
rz(3.020394332439658) q[10];
ry(-1.1796742089714156) q[11];
rz(1.5596902306354758) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5571148034355493) q[0];
rz(-0.05070439671568111) q[0];
ry(-0.24678473869465106) q[1];
rz(-0.7705664184029599) q[1];
ry(0.43346597873061565) q[2];
rz(-3.1383495769534906) q[2];
ry(0.04048250857878344) q[3];
rz(1.70236230981483) q[3];
ry(-3.129257314689072) q[4];
rz(-0.8458588440358774) q[4];
ry(2.9025129713173174) q[5];
rz(-0.016451072876753403) q[5];
ry(1.5716030975972268) q[6];
rz(2.51193672631042) q[6];
ry(-0.0030069953633211194) q[7];
rz(1.7927652437639665) q[7];
ry(-1.7372259252486089) q[8];
rz(1.1165151178003416) q[8];
ry(1.5827993581457624) q[9];
rz(1.5444989816498251) q[9];
ry(3.131121628639654) q[10];
rz(-0.12650637828940783) q[10];
ry(-1.5506192686072744) q[11];
rz(-3.0740071646445823) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-2.0879654730019404) q[0];
rz(1.3484462238157713) q[0];
ry(2.5449617478635864) q[1];
rz(-1.8053156680416054) q[1];
ry(1.604402489471947) q[2];
rz(0.0026423723645354613) q[2];
ry(1.5699385589309303) q[3];
rz(0.35542618365282913) q[3];
ry(3.1409037901755896) q[4];
rz(2.860100269158074) q[4];
ry(-1.5384706212386572) q[5];
rz(-3.1408279370582313) q[5];
ry(3.141285794706676) q[6];
rz(-2.306169795251196) q[6];
ry(1.4037652584226015) q[7];
rz(0.6963617086359246) q[7];
ry(1.4501092980299914) q[8];
rz(-3.0854091736961684) q[8];
ry(2.8519990118141467) q[9];
rz(-1.5022518768190685) q[9];
ry(0.7111326611841067) q[10];
rz(1.6174615116408646) q[10];
ry(2.4105203723959296) q[11];
rz(1.6188822906413212) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.138161625745852) q[0];
rz(0.15481696271544482) q[0];
ry(-0.16821552478970592) q[1];
rz(2.783363965570735) q[1];
ry(-3.13974621654946) q[2];
rz(1.9761566195335245) q[2];
ry(-0.004674630299877691) q[3];
rz(-1.883355077711158) q[3];
ry(3.1353950087234614) q[4];
rz(-1.4194809233548022) q[4];
ry(-1.570678398581454) q[5];
rz(-3.123781143895347) q[5];
ry(0.003354931420519236) q[6];
rz(-2.2802698957236567) q[6];
ry(0.0001401544160220319) q[7];
rz(2.4433078794357543) q[7];
ry(0.2868494191219195) q[8];
rz(2.449404594613836) q[8];
ry(2.7853035024333774) q[9];
rz(-1.637120709753925) q[9];
ry(1.4566838264207211) q[10];
rz(3.1060734954773594) q[10];
ry(-1.5032520633052053) q[11];
rz(-2.8111378517582133) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.04250659053777639) q[0];
rz(-1.92865805536176) q[0];
ry(0.20432439789936296) q[1];
rz(-1.255187483621901) q[1];
ry(-0.002840059628025138) q[2];
rz(2.7836543169659236) q[2];
ry(-1.6264572492717635) q[3];
rz(0.025455231111814487) q[3];
ry(-1.5834707358176168) q[4];
rz(1.5760822653387985) q[4];
ry(3.1093075866534767) q[5];
rz(-1.5571570646464208) q[5];
ry(1.5798895938777635) q[6];
rz(0.0016132713501679776) q[6];
ry(-1.5704411658907358) q[7];
rz(-3.121916511603887) q[7];
ry(3.1309210027770784) q[8];
rz(2.4831781248537323) q[8];
ry(-1.4498939835754314) q[9];
rz(2.3389007815527942) q[9];
ry(2.948531067987978) q[10];
rz(1.4676112713867555) q[10];
ry(-1.4146306319039532) q[11];
rz(0.0005587519789775895) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.569396427883571) q[0];
rz(-0.06711241414236624) q[0];
ry(-1.5733928704269164) q[1];
rz(-2.345926008731943) q[1];
ry(-1.5714079621012629) q[2];
rz(-1.5745137667171303) q[2];
ry(1.5655458356323197) q[3];
rz(3.1242630920757697) q[3];
ry(-1.5409112781751757) q[4];
rz(0.2473843964982386) q[4];
ry(1.5689224034938505) q[5];
rz(1.6555585619859354) q[5];
ry(2.7072505298560823) q[6];
rz(-6.497548056572099e-05) q[6];
ry(-0.4375199651334096) q[7];
rz(-2.605077169056342) q[7];
ry(-3.1142895150822296) q[8];
rz(-1.0190851453927487) q[8];
ry(-3.133206181067829) q[9];
rz(-0.33566148279357844) q[9];
ry(-0.07476246333694103) q[10];
rz(0.04933238960439735) q[10];
ry(-0.017405292200662892) q[11];
rz(0.8022350214463438) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.011890863409971253) q[0];
rz(-3.0683284950499434) q[0];
ry(-0.0007759795042909801) q[1];
rz(2.3603768523754884) q[1];
ry(1.5704529674697527) q[2];
rz(1.568915264983712) q[2];
ry(2.7854657242960053) q[3];
rz(-1.5828066937761214) q[3];
ry(-0.06977037264273457) q[4];
rz(1.325683806421333) q[4];
ry(3.139628671875088) q[5];
rz(0.08390707623261484) q[5];
ry(1.2400346562145523) q[6];
rz(-3.1156179655632172) q[6];
ry(0.016509368809945357) q[7];
rz(-1.1850809851753412) q[7];
ry(-1.827814279955045) q[8];
rz(0.1569889182282962) q[8];
ry(1.7276718366005424) q[9];
rz(-0.11095539359066567) q[9];
ry(1.5094009472091603) q[10];
rz(-0.1832579979368498) q[10];
ry(-0.24637337665879855) q[11];
rz(1.7551830252711715) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5731496646370797) q[0];
rz(-1.5698399599947281) q[0];
ry(1.5600091046804332) q[1];
rz(0.00131411406060834) q[1];
ry(1.6164142935486399) q[2];
rz(-1.6190722936906428) q[2];
ry(-1.5729014153642373) q[3];
rz(3.128078201698661) q[3];
ry(1.5711369337463976) q[4];
rz(2.8303613975999027) q[4];
ry(1.5705604142253653) q[5];
rz(0.18463051656892304) q[5];
ry(0.10477842230435763) q[6];
rz(1.7459516257764571) q[6];
ry(-3.017004662476195) q[7];
rz(0.9413954529841215) q[7];
ry(1.5642161989801346) q[8];
rz(1.6403956932073898) q[8];
ry(-1.5659918736899958) q[9];
rz(-1.5747955908955928) q[9];
ry(-3.0758154258914883) q[10];
rz(0.26745184613538364) q[10];
ry(-3.1381074364455652) q[11];
rz(1.0101012356440293) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.7362778960079306) q[0];
rz(-1.5059591809959976) q[0];
ry(-1.572421993259714) q[1];
rz(-0.0024973348540600075) q[1];
ry(-1.572210229258837) q[2];
rz(3.1398104213644156) q[2];
ry(-1.5766231258579584) q[3];
rz(-2.986237374742819) q[3];
ry(3.14076797194911) q[4];
rz(-1.8862096326435351) q[4];
ry(-0.00033711400006186665) q[5];
rz(2.934019653916619) q[5];
ry(-0.0027100652893992105) q[6];
rz(-1.3594275015210708) q[6];
ry(-1.7082588124573823) q[7];
rz(1.528757999564941) q[7];
ry(-3.0428979949342527) q[8];
rz(-1.4953793695991298) q[8];
ry(0.5112484086656032) q[9];
rz(-1.5611309073647415) q[9];
ry(-3.1364603131819404) q[10];
rz(0.9476925548451002) q[10];
ry(1.567346188013367) q[11];
rz(-0.002484023376513988) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(3.139944478141025) q[0];
rz(0.30508070916368) q[0];
ry(1.575799054110811) q[1];
rz(1.4443758635379877) q[1];
ry(-1.6171467750339978) q[2];
rz(1.5729503264889297) q[2];
ry(-3.1143248228527582) q[3];
rz(1.7262660810222314) q[3];
ry(0.5297328353380593) q[4];
rz(0.08140528776730617) q[4];
ry(1.5714755331262273) q[5];
rz(-0.3915964940038944) q[5];
ry(3.1398098807371815) q[6];
rz(0.08608579836307138) q[6];
ry(1.5744546027798663) q[7];
rz(-0.22257630687293967) q[7];
ry(1.5697294300730862) q[8];
rz(3.141289592235241) q[8];
ry(1.5697313817022822) q[9];
rz(2.911105569390614) q[9];
ry(0.01899166398905387) q[10];
rz(-1.8724831308058727) q[10];
ry(-1.542342477279352) q[11];
rz(-3.1319588859132206) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.004136236360605672) q[0];
rz(-2.446066999601937) q[0];
ry(3.139485395941943) q[1];
rz(-1.6984171277570903) q[1];
ry(1.5698126935749128) q[2];
rz(-1.6514441932000672) q[2];
ry(-1.5704814771551547) q[3];
rz(0.8459229211784897) q[3];
ry(-3.134730700930543) q[4];
rz(0.7232566317116659) q[4];
ry(-3.1414912582797303) q[5];
rz(-1.9573124339920451) q[5];
ry(9.40328922022167e-06) q[6];
rz(2.3870521569914334) q[6];
ry(-0.00018413345992565708) q[7];
rz(-2.709036106143938) q[7];
ry(-1.5704107964942766) q[8];
rz(-0.001186488278333897) q[8];
ry(0.001122434408932153) q[9];
rz(-1.3407528235910302) q[9];
ry(-1.5644309053799468) q[10];
rz(1.779960050492213) q[10];
ry(2.0697208481192533) q[11];
rz(-1.642381714227818) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.0002130948276652944) q[0];
rz(-2.504412638491028) q[0];
ry(1.5673294584933295) q[1];
rz(0.27415789997938317) q[1];
ry(-3.113461477711362) q[2];
rz(-2.0816773495340986) q[2];
ry(3.1391610779224894) q[3];
rz(-2.2932195846993357) q[3];
ry(3.0971871132252766) q[4];
rz(1.9651984246963676) q[4];
ry(-1.5306329570431358) q[5];
rz(-0.9203603955536938) q[5];
ry(0.019992292840045778) q[6];
rz(-2.185803635853723) q[6];
ry(-1.6561196705966728) q[7];
rz(3.0063148458838076) q[7];
ry(-1.5700710481888285) q[8];
rz(1.7584815046204163) q[8];
ry(1.5754442779509688) q[9];
rz(-1.8295475855517964) q[9];
ry(0.34281886466036493) q[10];
rz(2.9979390298431263) q[10];
ry(-1.5795966901421403) q[11];
rz(0.12509892429047761) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5857665605518572) q[0];
rz(1.569264544175276) q[0];
ry(-0.003316569290366722) q[1];
rz(-0.08445302383614005) q[1];
ry(-1.4340307924706908) q[2];
rz(-0.48694401384007185) q[2];
ry(1.1764290668085484) q[3];
rz(-0.0024526113194492482) q[3];
ry(1.4828189108584411e-05) q[4];
rz(0.25080021226790455) q[4];
ry(0.0003619418047436898) q[5];
rz(-2.2314215716973695) q[5];
ry(3.1409809785558167) q[6];
rz(-2.7153553385230627) q[6];
ry(-3.1411052003952173) q[7];
rz(3.013499953102433) q[7];
ry(0.03554410389605156) q[8];
rz(2.9540686051105975) q[8];
ry(3.1410402577749252) q[9];
rz(-0.2635440793052659) q[9];
ry(-1.5714549613486364) q[10];
rz(1.5713243409509356) q[10];
ry(-1.570270050518439) q[11];
rz(-3.1370576217342454) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5725805647460689) q[0];
rz(3.0709906403345633) q[0];
ry(-1.5700935283351918) q[1];
rz(-1.570193161393707) q[1];
ry(-0.2812775787767875) q[2];
rz(0.012140343091598796) q[2];
ry(1.570767502107741) q[3];
rz(-1.3841164513654052) q[3];
ry(1.571368589914636) q[4];
rz(-0.4639797384771007) q[4];
ry(-3.04502077199294) q[5];
rz(-2.463629573382006) q[5];
ry(3.130844333466476) q[6];
rz(1.6552093783654482) q[6];
ry(1.6547679432825761) q[7];
rz(-2.425219174386489) q[7];
ry(1.5730049734225457) q[8];
rz(-1.5699032961430004) q[8];
ry(3.138880250573199) q[9];
rz(-1.567041844982306) q[9];
ry(1.4707009103921078) q[10];
rz(1.9821223142705984) q[10];
ry(1.57036505113576) q[11];
rz(-2.8273949822534723) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.126994052052625) q[0];
rz(1.5040585197481056) q[0];
ry(-1.5713267713008339) q[1];
rz(1.5708895002191192) q[1];
ry(1.2745749822089287) q[2];
rz(-2.2336106139990317) q[2];
ry(-3.1375250568090283) q[3];
rz(-2.599726548145081) q[3];
ry(0.00011509713804880306) q[4];
rz(-1.1980777505867026) q[4];
ry(3.141459028157573) q[5];
rz(2.2593134477968406) q[5];
ry(0.003232914987783928) q[6];
rz(-2.6737308326061244) q[6];
ry(-0.00104027091039774) q[7];
rz(-0.9619833441502781) q[7];
ry(1.571426931827279) q[8];
rz(1.6758439425937581) q[8];
ry(-1.099828512471632) q[9];
rz(-0.2764978394945654) q[9];
ry(-3.1408129167471817) q[10];
rz(1.9762377466319825) q[10];
ry(3.139824760672385) q[11];
rz(1.786266882984571) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5769266458367477) q[0];
rz(3.029927080056802) q[0];
ry(1.570534280219957) q[1];
rz(-0.25745483961099325) q[1];
ry(-0.0011886471381732377) q[2];
rz(-2.414669209478003) q[2];
ry(-3.083549363724424) q[3];
rz(1.5990604681653466) q[3];
ry(0.0008255095984601861) q[4];
rz(-2.935400010948093) q[4];
ry(1.5734469484027402) q[5];
rz(2.854531629779847) q[5];
ry(-1.5702514450680436) q[6];
rz(2.264712644996628) q[6];
ry(0.00830478270530758) q[7];
rz(0.10419244721028954) q[7];
ry(0.004537814637363864) q[8];
rz(2.1521657505411698) q[8];
ry(-3.1366889473071407) q[9];
rz(2.509609186164681) q[9];
ry(-1.5711102656630258) q[10];
rz(-0.888961388153493) q[10];
ry(-1.5699062483838733) q[11];
rz(2.7827137594661018) q[11];