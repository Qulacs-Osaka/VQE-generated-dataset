OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.764324920382501) q[0];
ry(2.328179152134817) q[1];
cx q[0],q[1];
ry(1.455667754143416) q[0];
ry(0.3456788512346288) q[1];
cx q[0],q[1];
ry(1.8651557116666038) q[2];
ry(-3.0369777681210364) q[3];
cx q[2],q[3];
ry(-2.801624549321666) q[2];
ry(2.5999401911685642) q[3];
cx q[2],q[3];
ry(2.750671986277252) q[4];
ry(2.3516867330111535) q[5];
cx q[4],q[5];
ry(-2.2574490023811) q[4];
ry(-1.3126125646624622) q[5];
cx q[4],q[5];
ry(0.984103890359326) q[6];
ry(0.8022187695440569) q[7];
cx q[6],q[7];
ry(1.4700000239083337) q[6];
ry(2.930169612756742) q[7];
cx q[6],q[7];
ry(2.9850408168909874) q[0];
ry(-2.775996807879738) q[2];
cx q[0],q[2];
ry(1.4194105107752417) q[0];
ry(-2.661132361230099) q[2];
cx q[0],q[2];
ry(-1.4534251257676631) q[2];
ry(2.464210393296349) q[4];
cx q[2],q[4];
ry(1.3286327484249998) q[2];
ry(2.914593833314779) q[4];
cx q[2],q[4];
ry(-1.3425264789400355) q[4];
ry(0.012869226290067814) q[6];
cx q[4],q[6];
ry(2.7772312562445225) q[4];
ry(0.6670679971298306) q[6];
cx q[4],q[6];
ry(-1.2407792750506559) q[1];
ry(2.750301704602285) q[3];
cx q[1],q[3];
ry(-2.973600812466394) q[1];
ry(2.085497297484193) q[3];
cx q[1],q[3];
ry(1.5273088704484676) q[3];
ry(1.086659703618957) q[5];
cx q[3],q[5];
ry(-0.33356384487963364) q[3];
ry(1.036903518530889) q[5];
cx q[3],q[5];
ry(-2.7713114373022676) q[5];
ry(-0.9318994315298514) q[7];
cx q[5],q[7];
ry(1.1751885874862324) q[5];
ry(2.457929659775398) q[7];
cx q[5],q[7];
ry(2.953121944846098) q[0];
ry(0.055871127596517155) q[1];
cx q[0],q[1];
ry(-2.1376070105073444) q[0];
ry(2.919372934252843) q[1];
cx q[0],q[1];
ry(-1.2586337768493514) q[2];
ry(-2.972018545574197) q[3];
cx q[2],q[3];
ry(1.4518919636315113) q[2];
ry(0.8823027089310029) q[3];
cx q[2],q[3];
ry(-0.10319510781455588) q[4];
ry(1.5744192443376201) q[5];
cx q[4],q[5];
ry(2.214892334320914) q[4];
ry(-0.6918494789516663) q[5];
cx q[4],q[5];
ry(1.798363856800252) q[6];
ry(-0.17714176355587719) q[7];
cx q[6],q[7];
ry(2.350977350198959) q[6];
ry(-2.7165934629584876) q[7];
cx q[6],q[7];
ry(-2.199408222284717) q[0];
ry(2.1044871007745556) q[2];
cx q[0],q[2];
ry(-2.706762636187734) q[0];
ry(-2.871800629018673) q[2];
cx q[0],q[2];
ry(1.6429931151642743) q[2];
ry(-1.4781515485107237) q[4];
cx q[2],q[4];
ry(1.4266293306198148) q[2];
ry(-2.8637965784153674) q[4];
cx q[2],q[4];
ry(2.3265399801810145) q[4];
ry(-0.6658170694348824) q[6];
cx q[4],q[6];
ry(-0.2776403002043056) q[4];
ry(0.7420013218788126) q[6];
cx q[4],q[6];
ry(1.720996641955234) q[1];
ry(0.24578199015536661) q[3];
cx q[1],q[3];
ry(-0.6242555577727416) q[1];
ry(-3.114205581387589) q[3];
cx q[1],q[3];
ry(0.3632468514940906) q[3];
ry(1.0807322707254694) q[5];
cx q[3],q[5];
ry(0.08365593894325762) q[3];
ry(1.3352951258426462) q[5];
cx q[3],q[5];
ry(-1.1450240375594811) q[5];
ry(-0.993535618918231) q[7];
cx q[5],q[7];
ry(-2.914033120144034) q[5];
ry(-1.5130531685210986) q[7];
cx q[5],q[7];
ry(-1.7376411299220296) q[0];
ry(-2.9053516061848157) q[1];
cx q[0],q[1];
ry(1.6169395514375766) q[0];
ry(2.5933992456787966) q[1];
cx q[0],q[1];
ry(-0.671199111923836) q[2];
ry(1.8292852850233512) q[3];
cx q[2],q[3];
ry(-0.10396485854195259) q[2];
ry(2.822413124041925) q[3];
cx q[2],q[3];
ry(2.041701313733692) q[4];
ry(-2.952970855502758) q[5];
cx q[4],q[5];
ry(0.9165297853597743) q[4];
ry(3.0665619244008178) q[5];
cx q[4],q[5];
ry(1.7563088722422484) q[6];
ry(-1.8663842322444573) q[7];
cx q[6],q[7];
ry(-1.8310044280367377) q[6];
ry(-1.7459908308847907) q[7];
cx q[6],q[7];
ry(1.8412222072362665) q[0];
ry(-0.13588275641276723) q[2];
cx q[0],q[2];
ry(-1.8688940896203858) q[0];
ry(0.29648080673904825) q[2];
cx q[0],q[2];
ry(-2.520459897821643) q[2];
ry(-1.720977799824875) q[4];
cx q[2],q[4];
ry(-0.8339698356137007) q[2];
ry(0.28907466194077003) q[4];
cx q[2],q[4];
ry(-2.275321836537471) q[4];
ry(-2.7219333191798967) q[6];
cx q[4],q[6];
ry(-2.147854262608781) q[4];
ry(-0.8558169661241649) q[6];
cx q[4],q[6];
ry(2.414868309368715) q[1];
ry(-1.437519197140328) q[3];
cx q[1],q[3];
ry(-2.5147255948531484) q[1];
ry(3.1398281014709037) q[3];
cx q[1],q[3];
ry(0.4637694939818955) q[3];
ry(0.0568498701166155) q[5];
cx q[3],q[5];
ry(2.166280864505711) q[3];
ry(-0.586243576795721) q[5];
cx q[3],q[5];
ry(0.4323796310472314) q[5];
ry(3.104331976024236) q[7];
cx q[5],q[7];
ry(1.05281473653437) q[5];
ry(1.2202831898800106) q[7];
cx q[5],q[7];
ry(-0.3288006178654806) q[0];
ry(-2.856261207848783) q[1];
cx q[0],q[1];
ry(-1.8584969740160053) q[0];
ry(1.2603306605723006) q[1];
cx q[0],q[1];
ry(0.6744658951775122) q[2];
ry(0.36735383623812096) q[3];
cx q[2],q[3];
ry(0.4637656939178303) q[2];
ry(-0.7468277022954766) q[3];
cx q[2],q[3];
ry(-2.0437249752454525) q[4];
ry(1.4690866574377002) q[5];
cx q[4],q[5];
ry(1.2777576906523187) q[4];
ry(-2.820874932193266) q[5];
cx q[4],q[5];
ry(1.361410154898609) q[6];
ry(0.6433375822434513) q[7];
cx q[6],q[7];
ry(3.09874495026832) q[6];
ry(0.5445370992972434) q[7];
cx q[6],q[7];
ry(-2.9866551170951636) q[0];
ry(0.9371938016691211) q[2];
cx q[0],q[2];
ry(1.1513246691088779) q[0];
ry(1.391527713756522) q[2];
cx q[0],q[2];
ry(-1.0264892951084548) q[2];
ry(-1.9144301859123996) q[4];
cx q[2],q[4];
ry(3.061452411128918) q[2];
ry(1.2993473399958262) q[4];
cx q[2],q[4];
ry(0.07164610560977813) q[4];
ry(0.6920787091251333) q[6];
cx q[4],q[6];
ry(-1.20367599665372) q[4];
ry(1.7907816481764314) q[6];
cx q[4],q[6];
ry(-0.806367507185554) q[1];
ry(-1.4899650655668033) q[3];
cx q[1],q[3];
ry(0.2717405733900868) q[1];
ry(-1.171644993377228) q[3];
cx q[1],q[3];
ry(0.8113901745192753) q[3];
ry(2.610620551721679) q[5];
cx q[3],q[5];
ry(-2.435084931904823) q[3];
ry(1.2259627968993625) q[5];
cx q[3],q[5];
ry(1.4534488122433966) q[5];
ry(2.3357029353057914) q[7];
cx q[5],q[7];
ry(-1.4032770880169778) q[5];
ry(3.05233593699214) q[7];
cx q[5],q[7];
ry(3.1319176470181445) q[0];
ry(-0.8949903521444931) q[1];
cx q[0],q[1];
ry(-0.43117115482474944) q[0];
ry(-1.6721060615577827) q[1];
cx q[0],q[1];
ry(-2.5180481159282153) q[2];
ry(-1.1570167162605713) q[3];
cx q[2],q[3];
ry(-2.811912617546361) q[2];
ry(-3.0695120136072567) q[3];
cx q[2],q[3];
ry(0.18642322872318678) q[4];
ry(-2.3914790871134604) q[5];
cx q[4],q[5];
ry(-1.8283803704444477) q[4];
ry(-0.5416553381478657) q[5];
cx q[4],q[5];
ry(-2.9839964693916095) q[6];
ry(2.842403160228831) q[7];
cx q[6],q[7];
ry(-1.080632848473412) q[6];
ry(2.4616046771942584) q[7];
cx q[6],q[7];
ry(-0.16131584121162248) q[0];
ry(-2.3665572617393633) q[2];
cx q[0],q[2];
ry(-2.454073244095247) q[0];
ry(0.33736604436075535) q[2];
cx q[0],q[2];
ry(1.6869114143521158) q[2];
ry(-2.3139288828418367) q[4];
cx q[2],q[4];
ry(-0.5727297008426966) q[2];
ry(-1.930872141585399) q[4];
cx q[2],q[4];
ry(0.18238006484266914) q[4];
ry(0.6292333333100881) q[6];
cx q[4],q[6];
ry(0.9740836233634438) q[4];
ry(0.32992935298907256) q[6];
cx q[4],q[6];
ry(-1.2033110447314415) q[1];
ry(2.7824897042463026) q[3];
cx q[1],q[3];
ry(-1.3071689385934944) q[1];
ry(-1.4689499827142118) q[3];
cx q[1],q[3];
ry(-1.1537051112137853) q[3];
ry(-1.2132842272097064) q[5];
cx q[3],q[5];
ry(-1.7095014193397446) q[3];
ry(-1.0828688301566167) q[5];
cx q[3],q[5];
ry(-0.3509216138235125) q[5];
ry(-2.018081563640563) q[7];
cx q[5],q[7];
ry(-1.5918010877589883) q[5];
ry(-2.6240642820441766) q[7];
cx q[5],q[7];
ry(2.4358117561364776) q[0];
ry(-1.5931332884960652) q[1];
cx q[0],q[1];
ry(-0.6842069595336548) q[0];
ry(0.9325073114337058) q[1];
cx q[0],q[1];
ry(-2.870872850862951) q[2];
ry(0.09050044892536224) q[3];
cx q[2],q[3];
ry(1.9357326012036982) q[2];
ry(0.9150720014127929) q[3];
cx q[2],q[3];
ry(1.345721391162545) q[4];
ry(2.784244086068009) q[5];
cx q[4],q[5];
ry(1.9630013697279898) q[4];
ry(0.6975943632991962) q[5];
cx q[4],q[5];
ry(-2.4563756305486386) q[6];
ry(0.04800593478017223) q[7];
cx q[6],q[7];
ry(1.2449937098998118) q[6];
ry(-2.354762653150896) q[7];
cx q[6],q[7];
ry(-0.6414671411089642) q[0];
ry(0.871945370302698) q[2];
cx q[0],q[2];
ry(-0.21556405208118334) q[0];
ry(-1.5061953931452967) q[2];
cx q[0],q[2];
ry(1.4004180380151086) q[2];
ry(-0.7114002291923294) q[4];
cx q[2],q[4];
ry(0.9721807897039714) q[2];
ry(1.5733188676178507) q[4];
cx q[2],q[4];
ry(0.6892376108106948) q[4];
ry(1.3325501686769918) q[6];
cx q[4],q[6];
ry(-0.28315202857969024) q[4];
ry(-1.4987625428495122) q[6];
cx q[4],q[6];
ry(-0.8488784708546316) q[1];
ry(0.36567825022223666) q[3];
cx q[1],q[3];
ry(1.669081225167236) q[1];
ry(-2.51963993684761) q[3];
cx q[1],q[3];
ry(2.5657729476298554) q[3];
ry(0.4573767983347983) q[5];
cx q[3],q[5];
ry(0.32094025492569894) q[3];
ry(2.9173776926751547) q[5];
cx q[3],q[5];
ry(-2.8085439618683488) q[5];
ry(1.8139502270223833) q[7];
cx q[5],q[7];
ry(-2.068759980861935) q[5];
ry(3.0250720611226267) q[7];
cx q[5],q[7];
ry(2.467149547224774) q[0];
ry(-1.524039912310399) q[1];
cx q[0],q[1];
ry(-0.34067476327668705) q[0];
ry(1.4460539232479117) q[1];
cx q[0],q[1];
ry(-0.62628248146319) q[2];
ry(3.047331494758271) q[3];
cx q[2],q[3];
ry(1.76791780298169) q[2];
ry(1.90033432796539) q[3];
cx q[2],q[3];
ry(-1.7342255674433118) q[4];
ry(2.8379308665552796) q[5];
cx q[4],q[5];
ry(-2.9195504186146723) q[4];
ry(-0.1265579514046884) q[5];
cx q[4],q[5];
ry(2.97228554450889) q[6];
ry(0.49471568693452816) q[7];
cx q[6],q[7];
ry(-0.5326703920494724) q[6];
ry(-1.5119484542938453) q[7];
cx q[6],q[7];
ry(-3.118265382190324) q[0];
ry(2.7298945085735005) q[2];
cx q[0],q[2];
ry(1.116861363454806) q[0];
ry(0.5798335771257364) q[2];
cx q[0],q[2];
ry(-0.4089783989051927) q[2];
ry(0.3052447100885314) q[4];
cx q[2],q[4];
ry(-1.0846386744727408) q[2];
ry(-1.5952240897784444) q[4];
cx q[2],q[4];
ry(-1.817489763039295) q[4];
ry(-1.5841569877006547) q[6];
cx q[4],q[6];
ry(-0.7490255303166933) q[4];
ry(1.3843556131600814) q[6];
cx q[4],q[6];
ry(1.9282683269955818) q[1];
ry(-0.8641559973866525) q[3];
cx q[1],q[3];
ry(-2.064346888948265) q[1];
ry(1.8369592650573583) q[3];
cx q[1],q[3];
ry(0.3018747202230618) q[3];
ry(0.7924575244576008) q[5];
cx q[3],q[5];
ry(-0.7287463288177904) q[3];
ry(0.23506660677853564) q[5];
cx q[3],q[5];
ry(0.014188886813633951) q[5];
ry(0.09153537748885654) q[7];
cx q[5],q[7];
ry(-1.4348953581165236) q[5];
ry(-3.02799274719733) q[7];
cx q[5],q[7];
ry(-2.1873398208573107) q[0];
ry(-1.5016697541705888) q[1];
cx q[0],q[1];
ry(-0.6457409579359146) q[0];
ry(-3.089090778223616) q[1];
cx q[0],q[1];
ry(2.3848977751576097) q[2];
ry(0.67931992483142) q[3];
cx q[2],q[3];
ry(1.7322125670173545) q[2];
ry(-2.039685887315871) q[3];
cx q[2],q[3];
ry(-0.06800211342779328) q[4];
ry(-1.7657214056327435) q[5];
cx q[4],q[5];
ry(3.1131231048261148) q[4];
ry(-0.3218104743739625) q[5];
cx q[4],q[5];
ry(1.095048555286142) q[6];
ry(2.964154823957085) q[7];
cx q[6],q[7];
ry(1.7955568948594496) q[6];
ry(-2.4627961048133242) q[7];
cx q[6],q[7];
ry(-1.5891140737388767) q[0];
ry(1.3406859711189543) q[2];
cx q[0],q[2];
ry(-0.7256638502452909) q[0];
ry(1.4774371742051633) q[2];
cx q[0],q[2];
ry(2.2595268407431814) q[2];
ry(1.9052081911726477) q[4];
cx q[2],q[4];
ry(3.010827378536738) q[2];
ry(0.16150904603616062) q[4];
cx q[2],q[4];
ry(-0.42946435076253575) q[4];
ry(2.6922491114400104) q[6];
cx q[4],q[6];
ry(-3.010458545124444) q[4];
ry(0.6231203925671334) q[6];
cx q[4],q[6];
ry(1.1629547289219777) q[1];
ry(-0.5212379854275503) q[3];
cx q[1],q[3];
ry(-1.2091920207410234) q[1];
ry(0.20159814096921647) q[3];
cx q[1],q[3];
ry(2.6718684094779204) q[3];
ry(-1.8763799752502135) q[5];
cx q[3],q[5];
ry(-1.0152599519883108) q[3];
ry(-1.9849623377643524) q[5];
cx q[3],q[5];
ry(-2.0279968017186816) q[5];
ry(2.1475615734815863) q[7];
cx q[5],q[7];
ry(0.636541235565745) q[5];
ry(-1.0921156125792253) q[7];
cx q[5],q[7];
ry(2.166086837223925) q[0];
ry(-2.5817100832099467) q[1];
cx q[0],q[1];
ry(1.5760538466183345) q[0];
ry(1.9329464793141096) q[1];
cx q[0],q[1];
ry(-0.9192656504058121) q[2];
ry(1.5791132024474184) q[3];
cx q[2],q[3];
ry(0.9713257824728679) q[2];
ry(-0.7267997109678177) q[3];
cx q[2],q[3];
ry(-2.1707611620801535) q[4];
ry(2.9533969845573007) q[5];
cx q[4],q[5];
ry(-1.8293163542071422) q[4];
ry(0.2399227788088069) q[5];
cx q[4],q[5];
ry(2.844923268755207) q[6];
ry(2.538949778648362) q[7];
cx q[6],q[7];
ry(-0.6780533000079192) q[6];
ry(-2.7176463861558826) q[7];
cx q[6],q[7];
ry(0.812680583332048) q[0];
ry(0.5163397263043504) q[2];
cx q[0],q[2];
ry(2.8696310419475) q[0];
ry(1.189365277119962) q[2];
cx q[0],q[2];
ry(-2.6938443144256756) q[2];
ry(-0.7834981109318742) q[4];
cx q[2],q[4];
ry(1.3101880430376367) q[2];
ry(-2.9059633799597044) q[4];
cx q[2],q[4];
ry(1.2270202638908092) q[4];
ry(-2.0951684526642897) q[6];
cx q[4],q[6];
ry(-0.7190381582850023) q[4];
ry(-3.1333939751356366) q[6];
cx q[4],q[6];
ry(-2.7220207504867036) q[1];
ry(-0.3426818270023445) q[3];
cx q[1],q[3];
ry(1.9043293532738925) q[1];
ry(1.1651709020254186) q[3];
cx q[1],q[3];
ry(2.4923809673899884) q[3];
ry(-0.3746258754076903) q[5];
cx q[3],q[5];
ry(0.7591709743503254) q[3];
ry(0.9080242888519109) q[5];
cx q[3],q[5];
ry(-0.38325326657477365) q[5];
ry(-0.4235588855656358) q[7];
cx q[5],q[7];
ry(2.478833262649713) q[5];
ry(-1.976830756199935) q[7];
cx q[5],q[7];
ry(-2.6684347366523657) q[0];
ry(0.855085812989322) q[1];
cx q[0],q[1];
ry(1.7336454055511872) q[0];
ry(-2.8884248059844415) q[1];
cx q[0],q[1];
ry(0.08409219826234908) q[2];
ry(1.046400761666207) q[3];
cx q[2],q[3];
ry(0.9071260642019201) q[2];
ry(-0.27496904382018555) q[3];
cx q[2],q[3];
ry(1.8171905803076367) q[4];
ry(0.4344616481645715) q[5];
cx q[4],q[5];
ry(2.8468109895671008) q[4];
ry(-0.9449886742192556) q[5];
cx q[4],q[5];
ry(1.197350684817073) q[6];
ry(1.2863183242902636) q[7];
cx q[6],q[7];
ry(-2.633893271986816) q[6];
ry(1.3702943929773017) q[7];
cx q[6],q[7];
ry(1.9185683754607252) q[0];
ry(0.12415663839046509) q[2];
cx q[0],q[2];
ry(-0.7554645499590482) q[0];
ry(0.14785357726772652) q[2];
cx q[0],q[2];
ry(-0.32588762315663505) q[2];
ry(-2.6572810413795573) q[4];
cx q[2],q[4];
ry(-2.158101515331797) q[2];
ry(0.07753697350366644) q[4];
cx q[2],q[4];
ry(1.265248205996448) q[4];
ry(1.103347632982187) q[6];
cx q[4],q[6];
ry(1.58003224927139) q[4];
ry(-1.1331403076573556) q[6];
cx q[4],q[6];
ry(1.092437200207571) q[1];
ry(0.315027326761636) q[3];
cx q[1],q[3];
ry(0.9031889297555699) q[1];
ry(1.265589949384865) q[3];
cx q[1],q[3];
ry(0.6423328570404427) q[3];
ry(-1.46099336105231) q[5];
cx q[3],q[5];
ry(1.2849322095450768) q[3];
ry(2.7956589232134417) q[5];
cx q[3],q[5];
ry(-0.14056057512741216) q[5];
ry(-2.1434983554228935) q[7];
cx q[5],q[7];
ry(2.6372300590728597) q[5];
ry(-0.9131727902940501) q[7];
cx q[5],q[7];
ry(-1.6668292538469975) q[0];
ry(2.191566733953231) q[1];
cx q[0],q[1];
ry(-1.3457108530357103) q[0];
ry(-1.9494080356832235) q[1];
cx q[0],q[1];
ry(0.5514880100283268) q[2];
ry(-1.0509720807563945) q[3];
cx q[2],q[3];
ry(2.0813021490989256) q[2];
ry(-2.1813048477119636) q[3];
cx q[2],q[3];
ry(-2.073797612659531) q[4];
ry(1.0862574295715275) q[5];
cx q[4],q[5];
ry(2.406323509045018) q[4];
ry(0.23460721219837893) q[5];
cx q[4],q[5];
ry(0.1467563511632619) q[6];
ry(-0.07613132625757117) q[7];
cx q[6],q[7];
ry(0.13447454160543781) q[6];
ry(3.038942800222047) q[7];
cx q[6],q[7];
ry(1.2260731623163936) q[0];
ry(2.400945536527165) q[2];
cx q[0],q[2];
ry(0.7474042772594717) q[0];
ry(-0.3070513484439086) q[2];
cx q[0],q[2];
ry(0.5734876447344135) q[2];
ry(-1.494509106448481) q[4];
cx q[2],q[4];
ry(-1.4345353362887074) q[2];
ry(-2.0089746697575785) q[4];
cx q[2],q[4];
ry(3.0817310442079906) q[4];
ry(1.7632542108591578) q[6];
cx q[4],q[6];
ry(-0.7088582824729927) q[4];
ry(2.9184331762180546) q[6];
cx q[4],q[6];
ry(2.5974207558837334) q[1];
ry(0.7940420534367726) q[3];
cx q[1],q[3];
ry(1.3249418204668837) q[1];
ry(0.4055235972186884) q[3];
cx q[1],q[3];
ry(-0.5793840886325989) q[3];
ry(2.5568575487302265) q[5];
cx q[3],q[5];
ry(2.5130569804036718) q[3];
ry(0.08607601765359604) q[5];
cx q[3],q[5];
ry(-2.8278236932563665) q[5];
ry(3.060820521025805) q[7];
cx q[5],q[7];
ry(-1.6252567131621785) q[5];
ry(2.662163190730246) q[7];
cx q[5],q[7];
ry(0.980483181030996) q[0];
ry(-2.469348856900431) q[1];
cx q[0],q[1];
ry(1.3271727915357054) q[0];
ry(-2.161980855195986) q[1];
cx q[0],q[1];
ry(-0.7226084248873591) q[2];
ry(-0.32739949311897637) q[3];
cx q[2],q[3];
ry(1.4859458837086166) q[2];
ry(-0.49957232432219456) q[3];
cx q[2],q[3];
ry(0.6134167878906486) q[4];
ry(-1.2501818369106126) q[5];
cx q[4],q[5];
ry(-1.010030951529945) q[4];
ry(-1.258886355709702) q[5];
cx q[4],q[5];
ry(1.2350577889731902) q[6];
ry(0.9840947866747545) q[7];
cx q[6],q[7];
ry(-1.6505745293807204) q[6];
ry(-1.3575898391255772) q[7];
cx q[6],q[7];
ry(2.1630065442839745) q[0];
ry(2.762450791941677) q[2];
cx q[0],q[2];
ry(2.0668637583460847) q[0];
ry(1.7751301874880738) q[2];
cx q[0],q[2];
ry(1.2696315367100564) q[2];
ry(1.213407737195064) q[4];
cx q[2],q[4];
ry(-3.107717773247614) q[2];
ry(-1.401326294725444) q[4];
cx q[2],q[4];
ry(-1.4198543321882473) q[4];
ry(0.5748639787530907) q[6];
cx q[4],q[6];
ry(-2.3593549421159494) q[4];
ry(-1.1361050076352706) q[6];
cx q[4],q[6];
ry(-2.131045873033488) q[1];
ry(2.859367968324937) q[3];
cx q[1],q[3];
ry(-0.26484226756911106) q[1];
ry(-0.6699138850492172) q[3];
cx q[1],q[3];
ry(-2.201947836958232) q[3];
ry(2.618988776316581) q[5];
cx q[3],q[5];
ry(-1.375677804429741) q[3];
ry(-2.148004446708277) q[5];
cx q[3],q[5];
ry(-1.654016139721052) q[5];
ry(-2.727548410783327) q[7];
cx q[5],q[7];
ry(1.2906904211479047) q[5];
ry(1.595480626149767) q[7];
cx q[5],q[7];
ry(0.5375985094690581) q[0];
ry(2.4588019063530058) q[1];
cx q[0],q[1];
ry(-2.0562690160264) q[0];
ry(-1.6865062535839983) q[1];
cx q[0],q[1];
ry(-0.49503378478471927) q[2];
ry(0.6975573904727707) q[3];
cx q[2],q[3];
ry(0.053874511780646735) q[2];
ry(2.0438511131009243) q[3];
cx q[2],q[3];
ry(1.6570679922856213) q[4];
ry(0.4057894461953242) q[5];
cx q[4],q[5];
ry(-2.538772897950899) q[4];
ry(-1.0353435170244214) q[5];
cx q[4],q[5];
ry(-0.14753404191349923) q[6];
ry(-0.5228706272588111) q[7];
cx q[6],q[7];
ry(0.3957462296960071) q[6];
ry(1.8892572339779738) q[7];
cx q[6],q[7];
ry(-2.7216746885036622) q[0];
ry(-0.9591999746878875) q[2];
cx q[0],q[2];
ry(-1.0721270622886105) q[0];
ry(1.2985957472528094) q[2];
cx q[0],q[2];
ry(1.5759253672704574) q[2];
ry(2.519877361310205) q[4];
cx q[2],q[4];
ry(0.05673570797051428) q[2];
ry(-2.5086881799252785) q[4];
cx q[2],q[4];
ry(-2.310209275088879) q[4];
ry(-2.251052991932347) q[6];
cx q[4],q[6];
ry(-1.5273913388317317) q[4];
ry(-1.610309044679392) q[6];
cx q[4],q[6];
ry(-0.7561256418540374) q[1];
ry(-1.2380009103769671) q[3];
cx q[1],q[3];
ry(2.4015833138533154) q[1];
ry(2.840075995422693) q[3];
cx q[1],q[3];
ry(-1.2338934213197221) q[3];
ry(-0.8698156198278734) q[5];
cx q[3],q[5];
ry(1.281020362888295) q[3];
ry(2.356609172875951) q[5];
cx q[3],q[5];
ry(-1.0426690409668276) q[5];
ry(1.658987232280987) q[7];
cx q[5],q[7];
ry(0.024780479362307872) q[5];
ry(1.8249988057955353) q[7];
cx q[5],q[7];
ry(-1.607173352123488) q[0];
ry(-2.5187859019087733) q[1];
cx q[0],q[1];
ry(-0.9066159934485204) q[0];
ry(0.13053317051347424) q[1];
cx q[0],q[1];
ry(-2.054161315016261) q[2];
ry(-1.2792362209261192) q[3];
cx q[2],q[3];
ry(-2.24406878804679) q[2];
ry(-1.2929689133137254) q[3];
cx q[2],q[3];
ry(0.6233600386917979) q[4];
ry(-0.49759390694309413) q[5];
cx q[4],q[5];
ry(0.32634718459126777) q[4];
ry(-3.0799447066429337) q[5];
cx q[4],q[5];
ry(0.6530436976700199) q[6];
ry(2.258893978520051) q[7];
cx q[6],q[7];
ry(-2.8829019794413964) q[6];
ry(1.6664178315761156) q[7];
cx q[6],q[7];
ry(0.2700842858324212) q[0];
ry(1.1100084336188738) q[2];
cx q[0],q[2];
ry(-0.4182660847516689) q[0];
ry(1.2861531885325093) q[2];
cx q[0],q[2];
ry(-2.210469259247044) q[2];
ry(0.8924916645001439) q[4];
cx q[2],q[4];
ry(-0.36084603700215556) q[2];
ry(1.934843674425844) q[4];
cx q[2],q[4];
ry(1.6973927514665699) q[4];
ry(0.807614693573762) q[6];
cx q[4],q[6];
ry(2.4703964403798033) q[4];
ry(-1.1770960751494064) q[6];
cx q[4],q[6];
ry(1.1831288846375279) q[1];
ry(0.6426714076819978) q[3];
cx q[1],q[3];
ry(-2.9017482786169126) q[1];
ry(-0.3200573563408036) q[3];
cx q[1],q[3];
ry(-1.2655656531329547) q[3];
ry(0.3700204474663416) q[5];
cx q[3],q[5];
ry(-3.052342057756726) q[3];
ry(2.417777343962108) q[5];
cx q[3],q[5];
ry(0.9598817584707735) q[5];
ry(-1.0864315744952924) q[7];
cx q[5],q[7];
ry(2.6512739120401267) q[5];
ry(2.159870205030334) q[7];
cx q[5],q[7];
ry(2.310218280771584) q[0];
ry(-2.8493313779376575) q[1];
cx q[0],q[1];
ry(-2.9787419756617375) q[0];
ry(-1.903336656272477) q[1];
cx q[0],q[1];
ry(0.6391804141482256) q[2];
ry(-3.0197638072058908) q[3];
cx q[2],q[3];
ry(2.5835592997546994) q[2];
ry(2.054999521771461) q[3];
cx q[2],q[3];
ry(1.0498105471043604) q[4];
ry(0.8461971408478975) q[5];
cx q[4],q[5];
ry(2.7538329420928767) q[4];
ry(-2.936576056654936) q[5];
cx q[4],q[5];
ry(0.9902122769618247) q[6];
ry(2.2965321089263493) q[7];
cx q[6],q[7];
ry(-1.718845860470875) q[6];
ry(-1.728676854171515) q[7];
cx q[6],q[7];
ry(2.9906954704515005) q[0];
ry(0.4567963255026572) q[2];
cx q[0],q[2];
ry(1.8892591686490856) q[0];
ry(-0.9445193790028019) q[2];
cx q[0],q[2];
ry(-0.30336309198472716) q[2];
ry(-1.5160648744386185) q[4];
cx q[2],q[4];
ry(0.8783747397151176) q[2];
ry(3.13399734013409) q[4];
cx q[2],q[4];
ry(-2.5551640553096453) q[4];
ry(-1.080415177213517) q[6];
cx q[4],q[6];
ry(1.85751678555755) q[4];
ry(-2.4426899766192465) q[6];
cx q[4],q[6];
ry(0.5954623912650234) q[1];
ry(0.4849765502481053) q[3];
cx q[1],q[3];
ry(-0.21494846311816482) q[1];
ry(0.3867067990670352) q[3];
cx q[1],q[3];
ry(-0.4042472235227282) q[3];
ry(-2.8456126958831875) q[5];
cx q[3],q[5];
ry(-0.8963052639456448) q[3];
ry(-2.620097276389474) q[5];
cx q[3],q[5];
ry(0.9575183209704301) q[5];
ry(1.432535701194267) q[7];
cx q[5],q[7];
ry(-0.6836402109979486) q[5];
ry(-0.05683493743644075) q[7];
cx q[5],q[7];
ry(-0.4055085629824741) q[0];
ry(1.4724182299314483) q[1];
cx q[0],q[1];
ry(-0.8634195260718052) q[0];
ry(-1.2466313333704306) q[1];
cx q[0],q[1];
ry(2.833017446960949) q[2];
ry(-1.1487506641019367) q[3];
cx q[2],q[3];
ry(-2.415561453601571) q[2];
ry(1.4387597147166424) q[3];
cx q[2],q[3];
ry(-0.15708803650298012) q[4];
ry(-1.7091316489003292) q[5];
cx q[4],q[5];
ry(2.877513738825443) q[4];
ry(0.6421218735958174) q[5];
cx q[4],q[5];
ry(-2.1764048822181916) q[6];
ry(-1.148001578589973) q[7];
cx q[6],q[7];
ry(2.172390319700223) q[6];
ry(1.347907468357569) q[7];
cx q[6],q[7];
ry(-1.8675528056949195) q[0];
ry(1.8042544116094126) q[2];
cx q[0],q[2];
ry(2.4762274636074486) q[0];
ry(-0.6694723041865567) q[2];
cx q[0],q[2];
ry(-2.410939783859967) q[2];
ry(0.4107822816135642) q[4];
cx q[2],q[4];
ry(-0.6318687544294459) q[2];
ry(-1.0760335616890433) q[4];
cx q[2],q[4];
ry(-0.8299342140364718) q[4];
ry(-1.85921314910503) q[6];
cx q[4],q[6];
ry(-0.8001897470560961) q[4];
ry(-0.07230546264378139) q[6];
cx q[4],q[6];
ry(2.6687906974280313) q[1];
ry(2.1666377139471478) q[3];
cx q[1],q[3];
ry(-1.4126332430385369) q[1];
ry(2.0028722882985566) q[3];
cx q[1],q[3];
ry(2.299039011814612) q[3];
ry(-1.5421931009025647) q[5];
cx q[3],q[5];
ry(2.0042009636813978) q[3];
ry(1.0560794877659443) q[5];
cx q[3],q[5];
ry(-0.2326192811452135) q[5];
ry(2.2161505203802387) q[7];
cx q[5],q[7];
ry(2.728625106084733) q[5];
ry(-2.4435101865711726) q[7];
cx q[5],q[7];
ry(-1.8922514419293646) q[0];
ry(-0.7057462549446133) q[1];
cx q[0],q[1];
ry(-1.9582548162439366) q[0];
ry(-3.0681344306568596) q[1];
cx q[0],q[1];
ry(-0.19543682971579823) q[2];
ry(3.0363832123633356) q[3];
cx q[2],q[3];
ry(0.725034994464246) q[2];
ry(2.0356612630178708) q[3];
cx q[2],q[3];
ry(-0.14364429651282506) q[4];
ry(1.0382583339725153) q[5];
cx q[4],q[5];
ry(-0.33438470994999125) q[4];
ry(2.911016621581533) q[5];
cx q[4],q[5];
ry(1.1628816938321986) q[6];
ry(0.9665965147477787) q[7];
cx q[6],q[7];
ry(1.4933439520429692) q[6];
ry(-2.5411477285093893) q[7];
cx q[6],q[7];
ry(-0.0513720024179323) q[0];
ry(-1.1599217522716934) q[2];
cx q[0],q[2];
ry(-2.9682353462382327) q[0];
ry(-0.5540349212864601) q[2];
cx q[0],q[2];
ry(1.001165254762353) q[2];
ry(2.6524707144628143) q[4];
cx q[2],q[4];
ry(1.246683723340266) q[2];
ry(-3.1401076680721176) q[4];
cx q[2],q[4];
ry(1.5589200239474987) q[4];
ry(1.821726494247925) q[6];
cx q[4],q[6];
ry(-0.017767456318423174) q[4];
ry(-2.3675316868803065) q[6];
cx q[4],q[6];
ry(-2.2434931505724096) q[1];
ry(-1.3084456809958371) q[3];
cx q[1],q[3];
ry(-0.9951036057343652) q[1];
ry(-1.0272828626100328) q[3];
cx q[1],q[3];
ry(0.12178473929757948) q[3];
ry(0.3115163500344141) q[5];
cx q[3],q[5];
ry(1.2860515708804958) q[3];
ry(-3.0877726383346364) q[5];
cx q[3],q[5];
ry(1.9091446892264425) q[5];
ry(0.7812617464922056) q[7];
cx q[5],q[7];
ry(-2.4215715532008373) q[5];
ry(1.5246858801580456) q[7];
cx q[5],q[7];
ry(1.0905066649438042) q[0];
ry(0.17962224321815068) q[1];
cx q[0],q[1];
ry(-0.8894814230485305) q[0];
ry(-2.478989283742506) q[1];
cx q[0],q[1];
ry(1.4834572783213789) q[2];
ry(0.3909145164855398) q[3];
cx q[2],q[3];
ry(0.8641265661118611) q[2];
ry(-1.1317391001247783) q[3];
cx q[2],q[3];
ry(0.3167140003557618) q[4];
ry(1.5958741159094563) q[5];
cx q[4],q[5];
ry(-2.634357099490469) q[4];
ry(-2.4099567158837503) q[5];
cx q[4],q[5];
ry(2.926932682302705) q[6];
ry(2.170839514483042) q[7];
cx q[6],q[7];
ry(-0.23123312304648636) q[6];
ry(0.15684495855609096) q[7];
cx q[6],q[7];
ry(2.960985615055956) q[0];
ry(1.811233988701289) q[2];
cx q[0],q[2];
ry(2.290504029502457) q[0];
ry(1.9986427382927996) q[2];
cx q[0],q[2];
ry(-2.4453029171042386) q[2];
ry(-0.4169891678330382) q[4];
cx q[2],q[4];
ry(0.7513926203154759) q[2];
ry(2.9830438663762884) q[4];
cx q[2],q[4];
ry(0.9774039812369022) q[4];
ry(-1.4483544293982027) q[6];
cx q[4],q[6];
ry(-1.0713254552372025) q[4];
ry(-0.9538160374877315) q[6];
cx q[4],q[6];
ry(-2.3229270139923353) q[1];
ry(-2.4415778977933185) q[3];
cx q[1],q[3];
ry(2.5868324764283637) q[1];
ry(-2.2221599912889545) q[3];
cx q[1],q[3];
ry(-2.3335907230922084) q[3];
ry(-2.622318746537811) q[5];
cx q[3],q[5];
ry(-0.6431540133882424) q[3];
ry(0.9873215901332957) q[5];
cx q[3],q[5];
ry(-3.035022625184513) q[5];
ry(1.832056389480317) q[7];
cx q[5],q[7];
ry(0.9654568344296931) q[5];
ry(0.09625971748920835) q[7];
cx q[5],q[7];
ry(1.1468163908560178) q[0];
ry(-1.7192763790546026) q[1];
cx q[0],q[1];
ry(0.6235609936335466) q[0];
ry(2.2185824665257963) q[1];
cx q[0],q[1];
ry(0.6666632873842007) q[2];
ry(0.0746295400621264) q[3];
cx q[2],q[3];
ry(-1.57539464981413) q[2];
ry(-3.078387444739641) q[3];
cx q[2],q[3];
ry(-1.140254924916797) q[4];
ry(0.8359207960814462) q[5];
cx q[4],q[5];
ry(-1.2841622441608118) q[4];
ry(-2.726604997778011) q[5];
cx q[4],q[5];
ry(0.07060523584591727) q[6];
ry(1.4251817691757045) q[7];
cx q[6],q[7];
ry(2.1925339907961714) q[6];
ry(0.045289123366826174) q[7];
cx q[6],q[7];
ry(0.4438026605794172) q[0];
ry(0.8129892488795325) q[2];
cx q[0],q[2];
ry(-0.4460231716930396) q[0];
ry(0.15761848032571835) q[2];
cx q[0],q[2];
ry(-1.6246264135611739) q[2];
ry(1.5771915769916007) q[4];
cx q[2],q[4];
ry(-0.6065106898834687) q[2];
ry(-2.6728556530694623) q[4];
cx q[2],q[4];
ry(-2.8573938642884693) q[4];
ry(1.8766518418384448) q[6];
cx q[4],q[6];
ry(1.06539944039623) q[4];
ry(1.716379665047639) q[6];
cx q[4],q[6];
ry(0.5222664571989446) q[1];
ry(2.1156748167505874) q[3];
cx q[1],q[3];
ry(-0.33149337267102236) q[1];
ry(-1.6959233720818307) q[3];
cx q[1],q[3];
ry(0.1791588955814337) q[3];
ry(1.3991603781230415) q[5];
cx q[3],q[5];
ry(1.3774762044664257) q[3];
ry(-0.6388433759086087) q[5];
cx q[3],q[5];
ry(-2.512253445655496) q[5];
ry(2.7000184964004155) q[7];
cx q[5],q[7];
ry(-1.465764860564354) q[5];
ry(-2.3641441127088374) q[7];
cx q[5],q[7];
ry(-1.091652358808629) q[0];
ry(-1.1710960108347725) q[1];
cx q[0],q[1];
ry(2.026841773738374) q[0];
ry(-0.646086079216845) q[1];
cx q[0],q[1];
ry(1.4094471909945487) q[2];
ry(-2.366415813745839) q[3];
cx q[2],q[3];
ry(1.0789937036844401) q[2];
ry(-1.1471791609127728) q[3];
cx q[2],q[3];
ry(-2.608925063677738) q[4];
ry(-0.47205460007228434) q[5];
cx q[4],q[5];
ry(2.518027950351443) q[4];
ry(-0.24985718640833973) q[5];
cx q[4],q[5];
ry(0.9465337327057769) q[6];
ry(-0.5012385321759716) q[7];
cx q[6],q[7];
ry(-1.342284314560766) q[6];
ry(0.3245640775177138) q[7];
cx q[6],q[7];
ry(-0.5178863310584244) q[0];
ry(1.656194840510203) q[2];
cx q[0],q[2];
ry(1.9383889831873686) q[0];
ry(2.554634507107322) q[2];
cx q[0],q[2];
ry(-2.5018349999652822) q[2];
ry(1.685445028125729) q[4];
cx q[2],q[4];
ry(-1.7657360607241313) q[2];
ry(1.193654747095033) q[4];
cx q[2],q[4];
ry(-0.20291620542844857) q[4];
ry(-2.343550500055026) q[6];
cx q[4],q[6];
ry(2.2527084245741253) q[4];
ry(-1.3266088337597397) q[6];
cx q[4],q[6];
ry(-0.4306004097062299) q[1];
ry(-2.2816426060396138) q[3];
cx q[1],q[3];
ry(1.6982558740898275) q[1];
ry(-3.0626046180602327) q[3];
cx q[1],q[3];
ry(-1.4610557406486153) q[3];
ry(-0.9330209706661416) q[5];
cx q[3],q[5];
ry(2.9362443382764063) q[3];
ry(-0.22475725302107025) q[5];
cx q[3],q[5];
ry(-2.9904930125444067) q[5];
ry(-1.9692827979698897) q[7];
cx q[5],q[7];
ry(0.24871840824118685) q[5];
ry(2.4517950801365243) q[7];
cx q[5],q[7];
ry(1.6247769606196318) q[0];
ry(1.1822311842149418) q[1];
cx q[0],q[1];
ry(1.6280752730090997) q[0];
ry(2.8787062068318727) q[1];
cx q[0],q[1];
ry(-1.6833255366181923) q[2];
ry(-0.32131822066573346) q[3];
cx q[2],q[3];
ry(-0.9872489733339529) q[2];
ry(0.3565741622477949) q[3];
cx q[2],q[3];
ry(1.325732246163776) q[4];
ry(3.0792405016082807) q[5];
cx q[4],q[5];
ry(-0.6910690227175545) q[4];
ry(0.055111784324602835) q[5];
cx q[4],q[5];
ry(2.6312253138644595) q[6];
ry(0.8439862342972423) q[7];
cx q[6],q[7];
ry(1.8947933894510967) q[6];
ry(0.2327867979001823) q[7];
cx q[6],q[7];
ry(-2.3488242365393153) q[0];
ry(-0.9786291949096901) q[2];
cx q[0],q[2];
ry(-0.6353916664407083) q[0];
ry(1.785319062401638) q[2];
cx q[0],q[2];
ry(-2.338099737370193) q[2];
ry(-1.3447734803617672) q[4];
cx q[2],q[4];
ry(0.6209664955978786) q[2];
ry(-0.3955778771805652) q[4];
cx q[2],q[4];
ry(-0.794613150014663) q[4];
ry(-0.4071177328576123) q[6];
cx q[4],q[6];
ry(0.84120324126563) q[4];
ry(-2.22847668929351) q[6];
cx q[4],q[6];
ry(0.3429948447469233) q[1];
ry(-1.3364284873710393) q[3];
cx q[1],q[3];
ry(0.9902810215162652) q[1];
ry(-0.0016654759653133143) q[3];
cx q[1],q[3];
ry(-0.5462497505003592) q[3];
ry(2.647626288821861) q[5];
cx q[3],q[5];
ry(-0.07895160408817681) q[3];
ry(-0.1355162106123974) q[5];
cx q[3],q[5];
ry(-1.1227014135908366) q[5];
ry(-0.065266886573764) q[7];
cx q[5],q[7];
ry(1.3857419157527051) q[5];
ry(-2.054214826932508) q[7];
cx q[5],q[7];
ry(3.011531959878793) q[0];
ry(2.6870613290935044) q[1];
cx q[0],q[1];
ry(-2.137982771493057) q[0];
ry(0.22877988772315277) q[1];
cx q[0],q[1];
ry(1.6946528194008987) q[2];
ry(-1.9622820535344347) q[3];
cx q[2],q[3];
ry(1.8600458259230568) q[2];
ry(0.8165740449830317) q[3];
cx q[2],q[3];
ry(2.652803505366683) q[4];
ry(-2.6526287978507255) q[5];
cx q[4],q[5];
ry(-1.8515610345255726) q[4];
ry(0.12838437332745578) q[5];
cx q[4],q[5];
ry(0.9985616269164767) q[6];
ry(-1.533973538111172) q[7];
cx q[6],q[7];
ry(0.04013050689524203) q[6];
ry(2.928502634870773) q[7];
cx q[6],q[7];
ry(2.5941036037368055) q[0];
ry(0.45667137013878384) q[2];
cx q[0],q[2];
ry(2.4925398031651063) q[0];
ry(2.2762944257850584) q[2];
cx q[0],q[2];
ry(0.10993409822795823) q[2];
ry(-1.5065119011730825) q[4];
cx q[2],q[4];
ry(1.3252969994354364) q[2];
ry(1.5787324113610652) q[4];
cx q[2],q[4];
ry(2.6315171222268097) q[4];
ry(1.7219195495786936) q[6];
cx q[4],q[6];
ry(3.0643971670220616) q[4];
ry(3.066584775099239) q[6];
cx q[4],q[6];
ry(-1.8900346706719144) q[1];
ry(-1.4324582021426187) q[3];
cx q[1],q[3];
ry(1.3844215412738086) q[1];
ry(-1.5498747420951025) q[3];
cx q[1],q[3];
ry(-2.0388245648766334) q[3];
ry(-0.9648099421950906) q[5];
cx q[3],q[5];
ry(-2.163215092592168) q[3];
ry(-2.1298163022154295) q[5];
cx q[3],q[5];
ry(-1.4511773294433865) q[5];
ry(-2.06060595594024) q[7];
cx q[5],q[7];
ry(-1.2266865437822823) q[5];
ry(2.2559823029055064) q[7];
cx q[5],q[7];
ry(-2.592142378808197) q[0];
ry(2.366894939555505) q[1];
cx q[0],q[1];
ry(-2.798779736449439) q[0];
ry(2.364994888747151) q[1];
cx q[0],q[1];
ry(-3.048547301402996) q[2];
ry(-1.7897411784046175) q[3];
cx q[2],q[3];
ry(-3.1363469994112245) q[2];
ry(2.4050195116522746) q[3];
cx q[2],q[3];
ry(2.1776867147690195) q[4];
ry(2.378722352951285) q[5];
cx q[4],q[5];
ry(-0.33050162433957553) q[4];
ry(-2.7165679442057105) q[5];
cx q[4],q[5];
ry(-1.816972460859586) q[6];
ry(0.520519286855234) q[7];
cx q[6],q[7];
ry(-1.467500984585688) q[6];
ry(-1.2544158080927934) q[7];
cx q[6],q[7];
ry(-0.48618677771562024) q[0];
ry(-0.31941365521539655) q[2];
cx q[0],q[2];
ry(2.03413995865109) q[0];
ry(-2.14982809691136) q[2];
cx q[0],q[2];
ry(-3.0604316301766743) q[2];
ry(-2.89682510405536) q[4];
cx q[2],q[4];
ry(0.5005913811615562) q[2];
ry(1.8542443901377441) q[4];
cx q[2],q[4];
ry(0.15411060771506746) q[4];
ry(1.99613703440223) q[6];
cx q[4],q[6];
ry(1.3783345764397232) q[4];
ry(0.9811705446932609) q[6];
cx q[4],q[6];
ry(-2.997608974737317) q[1];
ry(0.3838538160268753) q[3];
cx q[1],q[3];
ry(2.1882900536211753) q[1];
ry(-0.5354833990859471) q[3];
cx q[1],q[3];
ry(2.504728309744197) q[3];
ry(-2.007908127429592) q[5];
cx q[3],q[5];
ry(2.770131423222196) q[3];
ry(-0.8507926251153384) q[5];
cx q[3],q[5];
ry(0.37017001458526355) q[5];
ry(0.027611140971782966) q[7];
cx q[5],q[7];
ry(0.7646419983534438) q[5];
ry(-2.9792243619878804) q[7];
cx q[5],q[7];
ry(-1.8439621471254424) q[0];
ry(-0.7558034570017124) q[1];
cx q[0],q[1];
ry(-3.1242050969319455) q[0];
ry(2.8912845819756092) q[1];
cx q[0],q[1];
ry(-2.6108579753315384) q[2];
ry(-2.498928669103899) q[3];
cx q[2],q[3];
ry(1.5346509964568549) q[2];
ry(2.833817931056105) q[3];
cx q[2],q[3];
ry(0.6283908810814475) q[4];
ry(0.06061522770393424) q[5];
cx q[4],q[5];
ry(-0.8083879278561765) q[4];
ry(0.9353361778094023) q[5];
cx q[4],q[5];
ry(-0.11987764171973671) q[6];
ry(0.7905452068985586) q[7];
cx q[6],q[7];
ry(-2.251137353337639) q[6];
ry(2.94835599100055) q[7];
cx q[6],q[7];
ry(-2.1098003884490204) q[0];
ry(-0.5235494297599788) q[2];
cx q[0],q[2];
ry(-2.4710820663263027) q[0];
ry(2.456791602989837) q[2];
cx q[0],q[2];
ry(-0.1549661612355133) q[2];
ry(-1.0484224217387925) q[4];
cx q[2],q[4];
ry(1.5959265247429395) q[2];
ry(1.5505858842766282) q[4];
cx q[2],q[4];
ry(2.33135722774634) q[4];
ry(-0.3698143357891288) q[6];
cx q[4],q[6];
ry(0.9078986996940346) q[4];
ry(-3.045125236774574) q[6];
cx q[4],q[6];
ry(-0.45990372717800865) q[1];
ry(0.3485847150903179) q[3];
cx q[1],q[3];
ry(-1.5177357622268737) q[1];
ry(-2.3367341011829543) q[3];
cx q[1],q[3];
ry(1.1670144067580825) q[3];
ry(1.1452006780126145) q[5];
cx q[3],q[5];
ry(0.5647267829770479) q[3];
ry(-1.2217354881420364) q[5];
cx q[3],q[5];
ry(-0.23470749160188387) q[5];
ry(-2.30734647043761) q[7];
cx q[5],q[7];
ry(1.245788760246974) q[5];
ry(-1.7308913236688064) q[7];
cx q[5],q[7];
ry(-1.7568044204631557) q[0];
ry(-1.6487632269032702) q[1];
cx q[0],q[1];
ry(2.0720746572055706) q[0];
ry(-1.6428763134111897) q[1];
cx q[0],q[1];
ry(0.5835596855052254) q[2];
ry(-2.726513768914546) q[3];
cx q[2],q[3];
ry(0.7846865637504543) q[2];
ry(2.183622432313702) q[3];
cx q[2],q[3];
ry(-1.7864391875244703) q[4];
ry(-1.31445058246504) q[5];
cx q[4],q[5];
ry(-2.726653526613622) q[4];
ry(-2.931504977917854) q[5];
cx q[4],q[5];
ry(-3.073821074915017) q[6];
ry(-1.0418805702349152) q[7];
cx q[6],q[7];
ry(-0.664629057557609) q[6];
ry(1.8666712587101042) q[7];
cx q[6],q[7];
ry(-3.017597375703481) q[0];
ry(1.7603577170920177) q[2];
cx q[0],q[2];
ry(-1.5836291577307342) q[0];
ry(-1.6319500049470967) q[2];
cx q[0],q[2];
ry(-2.313764230147962) q[2];
ry(-3.1106175924471455) q[4];
cx q[2],q[4];
ry(-2.6804982361893637) q[2];
ry(0.1970684349736036) q[4];
cx q[2],q[4];
ry(0.11947495264957939) q[4];
ry(1.6463427503398977) q[6];
cx q[4],q[6];
ry(-2.977708576116459) q[4];
ry(0.41476670016688466) q[6];
cx q[4],q[6];
ry(0.5251007735378534) q[1];
ry(-2.7757495795482865) q[3];
cx q[1],q[3];
ry(-1.9571035046084753) q[1];
ry(-2.3622708363898703) q[3];
cx q[1],q[3];
ry(-2.968162805930453) q[3];
ry(-2.6887052489840175) q[5];
cx q[3],q[5];
ry(-2.4838105180379513) q[3];
ry(-2.412563964184888) q[5];
cx q[3],q[5];
ry(0.8913421019340146) q[5];
ry(-0.3591851486558033) q[7];
cx q[5],q[7];
ry(-1.718560334255004) q[5];
ry(0.14769496365764948) q[7];
cx q[5],q[7];
ry(0.43837054815215326) q[0];
ry(0.3701714243751555) q[1];
cx q[0],q[1];
ry(-0.5161086939839254) q[0];
ry(2.481327976178368) q[1];
cx q[0],q[1];
ry(-2.0359379076437287) q[2];
ry(1.1765165240532216) q[3];
cx q[2],q[3];
ry(0.0386810515373881) q[2];
ry(2.688273214806343) q[3];
cx q[2],q[3];
ry(1.3431992387982312) q[4];
ry(1.2148561577435029) q[5];
cx q[4],q[5];
ry(-1.220667660049803) q[4];
ry(-1.9694914824526337) q[5];
cx q[4],q[5];
ry(1.4423001427361912) q[6];
ry(-2.0427351463503154) q[7];
cx q[6],q[7];
ry(-0.7363439947927622) q[6];
ry(-1.4703909340449475) q[7];
cx q[6],q[7];
ry(-3.1236063606067046) q[0];
ry(1.6871430151337945) q[2];
cx q[0],q[2];
ry(-0.8333516224168713) q[0];
ry(-1.6194693751037674) q[2];
cx q[0],q[2];
ry(0.9989963109638792) q[2];
ry(0.2901004572980277) q[4];
cx q[2],q[4];
ry(0.365956508516609) q[2];
ry(1.4302469332269254) q[4];
cx q[2],q[4];
ry(-2.188215698335525) q[4];
ry(-1.285733950610723) q[6];
cx q[4],q[6];
ry(-3.097851756029272) q[4];
ry(-0.8255285803381387) q[6];
cx q[4],q[6];
ry(-0.5094748253983372) q[1];
ry(1.4617486639242785) q[3];
cx q[1],q[3];
ry(0.2801863335323356) q[1];
ry(1.0809972420570562) q[3];
cx q[1],q[3];
ry(2.4919399838087797) q[3];
ry(-2.8652402300808304) q[5];
cx q[3],q[5];
ry(1.642305946352226) q[3];
ry(-0.49704702402286927) q[5];
cx q[3],q[5];
ry(-2.004207790031912) q[5];
ry(-0.3523439934737222) q[7];
cx q[5],q[7];
ry(0.8967554224151444) q[5];
ry(3.1297453034882206) q[7];
cx q[5],q[7];
ry(-2.3917861120941133) q[0];
ry(2.356989541338988) q[1];
cx q[0],q[1];
ry(2.645107478378746) q[0];
ry(2.2970366967036893) q[1];
cx q[0],q[1];
ry(-2.7504214203425823) q[2];
ry(0.25221514044352483) q[3];
cx q[2],q[3];
ry(2.96897165998223) q[2];
ry(-0.7834526338101772) q[3];
cx q[2],q[3];
ry(-0.7357251966747487) q[4];
ry(3.0244201587111523) q[5];
cx q[4],q[5];
ry(-0.19339390361883516) q[4];
ry(-2.4095977955330055) q[5];
cx q[4],q[5];
ry(0.44458293447834835) q[6];
ry(-2.3751806628376912) q[7];
cx q[6],q[7];
ry(-0.13510773817840513) q[6];
ry(2.921243389598021) q[7];
cx q[6],q[7];
ry(-2.081134886502703) q[0];
ry(-1.8717802993901813) q[2];
cx q[0],q[2];
ry(-1.6231486066502825) q[0];
ry(-0.18890314385870344) q[2];
cx q[0],q[2];
ry(-0.6849087737453363) q[2];
ry(-2.9497553220167347) q[4];
cx q[2],q[4];
ry(-2.4623189555749976) q[2];
ry(0.5846240237516361) q[4];
cx q[2],q[4];
ry(0.29992373897493607) q[4];
ry(2.9770020048489667) q[6];
cx q[4],q[6];
ry(1.8553522571332715) q[4];
ry(0.1301183390422215) q[6];
cx q[4],q[6];
ry(2.8019657814066603) q[1];
ry(2.417700123424141) q[3];
cx q[1],q[3];
ry(1.0973170641720857) q[1];
ry(0.7486919859645056) q[3];
cx q[1],q[3];
ry(0.16629859728557805) q[3];
ry(2.201969042533512) q[5];
cx q[3],q[5];
ry(-2.432112296607642) q[3];
ry(-0.03506520842348374) q[5];
cx q[3],q[5];
ry(1.575641655323177) q[5];
ry(-1.5436997367423495) q[7];
cx q[5],q[7];
ry(-0.5755039966671793) q[5];
ry(-3.1387302304711633) q[7];
cx q[5],q[7];
ry(-0.6332288645336357) q[0];
ry(0.380128714208802) q[1];
ry(2.196224399176139) q[2];
ry(-2.201407409306691) q[3];
ry(-1.376193238470304) q[4];
ry(2.5675630388354684) q[5];
ry(-2.06788008646206) q[6];
ry(1.7375783778282536) q[7];