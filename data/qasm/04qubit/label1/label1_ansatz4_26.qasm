OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(3.1311096544049875) q[0];
rz(0.4383121122653799) q[0];
ry(1.6972033245197244) q[1];
rz(2.9665474343136786) q[1];
ry(2.4738571362029056) q[2];
rz(-2.9060855777158836) q[2];
ry(-2.1005433125070065) q[3];
rz(-0.7128960778914292) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3878746447626762) q[0];
rz(-0.5019680877428163) q[0];
ry(0.5893755286760332) q[1];
rz(1.7578217078129168) q[1];
ry(1.1827934297887395) q[2];
rz(3.053056887958739) q[2];
ry(2.0900440013460484) q[3];
rz(1.1282769501399537) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.042646807155056) q[0];
rz(-0.7097241506267578) q[0];
ry(-0.48719582739257017) q[1];
rz(-0.2072694172173475) q[1];
ry(2.0420690354353237) q[2];
rz(-2.905455029844195) q[2];
ry(-2.1216081665943616) q[3];
rz(-1.4989856478132526) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.37949177428415926) q[0];
rz(-0.6993429816103873) q[0];
ry(-1.6203268578909364) q[1];
rz(0.44191353111001774) q[1];
ry(2.318035066884527) q[2];
rz(-1.1774346104822417) q[2];
ry(0.9979475645642808) q[3];
rz(-2.919917898621115) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3973446258919786) q[0];
rz(0.6920044183369836) q[0];
ry(2.963888532143976) q[1];
rz(-1.6725828329766137) q[1];
ry(3.0240416051271994) q[2];
rz(2.3278484456585433) q[2];
ry(-0.3805988949654324) q[3];
rz(-0.20042973198699787) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6509863176544214) q[0];
rz(0.7520840992902418) q[0];
ry(-2.532008077352981) q[1];
rz(2.7633073532438206) q[1];
ry(-1.965738750513692) q[2];
rz(2.0869724734119846) q[2];
ry(-0.39733053484708325) q[3];
rz(-1.8853707822076595) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.8324118795930523) q[0];
rz(-0.8644716737022975) q[0];
ry(-2.778022264504683) q[1];
rz(-2.94185752327949) q[1];
ry(0.8807675862281232) q[2];
rz(3.1251623791541756) q[2];
ry(2.692137998045371) q[3];
rz(0.6585169626568456) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7200396136084717) q[0];
rz(3.082307424107677) q[0];
ry(1.4115845289880786) q[1];
rz(2.887770508998465) q[1];
ry(-2.229117613716352) q[2];
rz(1.3627720058990649) q[2];
ry(-1.4492908434144434) q[3];
rz(-2.8893512977277167) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.21665671774962011) q[0];
rz(-1.7161641639386107) q[0];
ry(1.7734167254749993) q[1];
rz(0.24425923189750254) q[1];
ry(0.5940183373499293) q[2];
rz(1.188734082801437) q[2];
ry(-2.7308656163857647) q[3];
rz(2.9757979509186083) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.034384813834978) q[0];
rz(1.3643865259839416) q[0];
ry(-1.2360546173714986) q[1];
rz(-0.3858770402461497) q[1];
ry(3.000275416920699) q[2];
rz(0.36472563262913593) q[2];
ry(-2.897788358562091) q[3];
rz(-0.5593679031125206) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.9925257361963287) q[0];
rz(-2.474077897284976) q[0];
ry(1.6434292216469992) q[1];
rz(3.080676635587433) q[1];
ry(2.8902787339972753) q[2];
rz(-2.9533233718155465) q[2];
ry(-2.2957498527648137) q[3];
rz(-2.539535508622346) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5222554244805053) q[0];
rz(-2.374536381386956) q[0];
ry(-2.4950884496866617) q[1];
rz(3.0430158761277912) q[1];
ry(-2.2655332496124867) q[2];
rz(2.292862260438844) q[2];
ry(-0.3435165812577065) q[3];
rz(-0.471623010729756) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.15761340190562387) q[0];
rz(-0.9280090692584364) q[0];
ry(2.4400727759914274) q[1];
rz(-2.424880297791606) q[1];
ry(-0.20895343791914975) q[2];
rz(0.8943741048290597) q[2];
ry(-2.9508657247946966) q[3];
rz(-2.4741886424757196) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.8499470415714324) q[0];
rz(0.8782926392451387) q[0];
ry(-2.771344910238942) q[1];
rz(0.5065675230852983) q[1];
ry(1.4479507896455044) q[2];
rz(1.2262119048558504) q[2];
ry(-1.847705114133107) q[3];
rz(0.06834107134829048) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.46508002813203225) q[0];
rz(-1.4683397688253574) q[0];
ry(-1.985694590405806) q[1];
rz(-1.203448035689778) q[1];
ry(2.4246202417536367) q[2];
rz(0.07516502230492285) q[2];
ry(2.7665398651033577) q[3];
rz(2.1211198062750585) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.37381527735457093) q[0];
rz(-0.8624737574125212) q[0];
ry(-2.4540791919343636) q[1];
rz(2.099207741909037) q[1];
ry(0.5295534349244376) q[2];
rz(-0.9472788613854451) q[2];
ry(-2.3367733252810576) q[3];
rz(-1.9684839464690231) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.8218240088487976) q[0];
rz(0.7026671443162656) q[0];
ry(2.5114575233822065) q[1];
rz(-0.6733087789158823) q[1];
ry(-1.8758256276253167) q[2];
rz(1.64051348760041) q[2];
ry(1.3873292289115953) q[3];
rz(-2.6321230840308933) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.1982877858167918) q[0];
rz(1.9196035171373786) q[0];
ry(-0.07183331206372401) q[1];
rz(0.2672725104793215) q[1];
ry(-2.192567819601499) q[2];
rz(-0.9039490270835024) q[2];
ry(-0.6902053414390652) q[3];
rz(-0.21226593140884412) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.9581428544646675) q[0];
rz(1.8151630242588057) q[0];
ry(-0.14481787053199807) q[1];
rz(0.44587337086605844) q[1];
ry(1.6705409153910973) q[2];
rz(-1.5868733249620819) q[2];
ry(0.7889371997158756) q[3];
rz(0.8879828630369647) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.2385084266404922) q[0];
rz(-2.9259845222397574) q[0];
ry(-1.774104009965824) q[1];
rz(1.161126745328883) q[1];
ry(-1.4872269761195382) q[2];
rz(0.44541704012163574) q[2];
ry(2.869143046443991) q[3];
rz(-2.9990622305784185) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5944231269776808) q[0];
rz(-2.3559675484437963) q[0];
ry(1.7428819135556193) q[1];
rz(-0.265052366378692) q[1];
ry(0.517063586724201) q[2];
rz(-2.4956987463138756) q[2];
ry(-2.800959414679503) q[3];
rz(1.4466020129887476) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.9787345741467375) q[0];
rz(0.40611051935583137) q[0];
ry(0.9349399215427548) q[1];
rz(-0.34684962743282277) q[1];
ry(2.4157795256835763) q[2];
rz(2.6700410996707733) q[2];
ry(1.6898458611248064) q[3];
rz(-2.783767860438148) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.698407795191991) q[0];
rz(2.571929870992998) q[0];
ry(-1.4393881957415706) q[1];
rz(0.9128254079254372) q[1];
ry(-0.13350594880345013) q[2];
rz(-1.8673372161794661) q[2];
ry(-1.4511286163774457) q[3];
rz(-0.37045777194956386) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.23506323178353491) q[0];
rz(-2.09890982040759) q[0];
ry(3.0821365898492177) q[1];
rz(-2.714666138954042) q[1];
ry(-2.321884844770367) q[2];
rz(-2.2688876069781854) q[2];
ry(-0.4133634734819234) q[3];
rz(0.660815062486237) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.977039283796895) q[0];
rz(0.5502442444113073) q[0];
ry(-1.1555587688507998) q[1];
rz(1.467108623372642) q[1];
ry(-0.4461181968992133) q[2];
rz(1.6841323491821303) q[2];
ry(-1.8931948843258342) q[3];
rz(-0.4117844578865221) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.409447686820406) q[0];
rz(-2.1862118888639257) q[0];
ry(-2.3998185292199543) q[1];
rz(1.3039998798350256) q[1];
ry(-0.5333330263083644) q[2];
rz(2.508347308713456) q[2];
ry(-2.764529046739186) q[3];
rz(-0.626740381636988) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.5856734708377712) q[0];
rz(-1.8250377795788646) q[0];
ry(-0.7373175350055873) q[1];
rz(-2.2059910590170064) q[1];
ry(-1.4297960460355637) q[2];
rz(-1.4898510703899834) q[2];
ry(-0.3425680454401281) q[3];
rz(-1.488521769357714) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.403259077998559) q[0];
rz(-0.39323882033990887) q[0];
ry(-1.9489258587129286) q[1];
rz(0.6893956920225103) q[1];
ry(2.842263478510025) q[2];
rz(0.24675514155569436) q[2];
ry(-2.294747840333332) q[3];
rz(-1.604480451605224) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.369199523602785) q[0];
rz(0.3626442700564461) q[0];
ry(2.506091225552237) q[1];
rz(-2.351777164234273) q[1];
ry(1.4698868484540535) q[2];
rz(2.2273263098113203) q[2];
ry(-0.0736140672535275) q[3];
rz(0.8894470341734904) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.6553269025700602) q[0];
rz(2.779177118694056) q[0];
ry(-2.7201865610427083) q[1];
rz(1.3986447410229914) q[1];
ry(2.6532944840342854) q[2];
rz(-1.4308134947284246) q[2];
ry(0.0952639208877409) q[3];
rz(-2.588700849727536) q[3];