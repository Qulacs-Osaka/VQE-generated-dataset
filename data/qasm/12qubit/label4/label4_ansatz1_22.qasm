OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.7869101233805313) q[0];
rz(-0.9921759035505885) q[0];
ry(-0.0705408238791457) q[1];
rz(-0.21689531244131446) q[1];
ry(2.8503986532391883) q[2];
rz(0.8651904058163162) q[2];
ry(0.07621867311872284) q[3];
rz(1.2573993107950503) q[3];
ry(-0.2736880623098079) q[4];
rz(-1.38490868271949) q[4];
ry(3.128040786851913) q[5];
rz(-1.529351598874829) q[5];
ry(-0.8924738300510633) q[6];
rz(2.6150550758218265) q[6];
ry(2.4651327779053998) q[7];
rz(-2.9605017708945227) q[7];
ry(2.3425118860764496) q[8];
rz(-3.133882860022734) q[8];
ry(0.021141537480972694) q[9];
rz(-2.436343837003309) q[9];
ry(1.5442277381274758) q[10];
rz(-1.579315600635831) q[10];
ry(-0.08734600200225985) q[11];
rz(-0.5008898999821296) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.3168819815591144) q[0];
rz(-0.5804166057274606) q[0];
ry(0.4377678452848964) q[1];
rz(-0.08622370828409665) q[1];
ry(1.8873229071480475) q[2];
rz(0.4689103023589913) q[2];
ry(1.4500407312495238) q[3];
rz(1.0522360745510673) q[3];
ry(0.11747846919065996) q[4];
rz(1.4444288479908227) q[4];
ry(-2.5130847374108347) q[5];
rz(1.8341108933605543) q[5];
ry(1.5733030373340848) q[6];
rz(2.9617236877581687) q[6];
ry(2.7457671283366247) q[7];
rz(-1.8482953232902875) q[7];
ry(-2.681279838637779) q[8];
rz(0.03225403439997798) q[8];
ry(0.005749695505334884) q[9];
rz(-0.7736122489392735) q[9];
ry(-2.893593938680155) q[10];
rz(-2.59968476871691) q[10];
ry(-1.6620058632481924) q[11];
rz(0.31565632641677605) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.0720781398633994) q[0];
rz(1.038443805533598) q[0];
ry(0.21378603646597724) q[1];
rz(-1.243849512502929) q[1];
ry(1.782592902658221) q[2];
rz(1.2773996947337887) q[2];
ry(0.7885139788454758) q[3];
rz(1.691360839212248) q[3];
ry(3.1173442789022587) q[4];
rz(1.5045358720739295) q[4];
ry(0.028544898613804826) q[5];
rz(3.0004253252975817) q[5];
ry(0.054972136269468036) q[6];
rz(-0.8401663725046733) q[6];
ry(-0.6870364008024935) q[7];
rz(-0.6621854758402934) q[7];
ry(0.4738546978514125) q[8];
rz(-0.6836188309832888) q[8];
ry(1.394421822026395) q[9];
rz(0.5901270118492441) q[9];
ry(-1.4120171282401328) q[10];
rz(-2.6164044521021292) q[10];
ry(-2.453024613430725) q[11];
rz(-0.7022048306173998) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.0395094107718608) q[0];
rz(-1.790121483967355) q[0];
ry(0.19327758586656818) q[1];
rz(-2.071003180501693) q[1];
ry(-0.3275606547075316) q[2];
rz(-1.9653752757379575) q[2];
ry(2.7200420217994874) q[3];
rz(2.107824600879665) q[3];
ry(-0.018459711504608154) q[4];
rz(-2.247190355748658) q[4];
ry(-1.461602295899036) q[5];
rz(-1.0482830264150858) q[5];
ry(2.0574011003861714) q[6];
rz(-0.6165669169189336) q[6];
ry(-2.531123385655827) q[7];
rz(-1.6172323411254779) q[7];
ry(1.3680553918748641) q[8];
rz(0.9938131430151471) q[8];
ry(-3.1129324282357365) q[9];
rz(3.053269870268421) q[9];
ry(0.22698749384948247) q[10];
rz(0.15546066175041565) q[10];
ry(2.0242440241018196) q[11];
rz(-0.9247514337904558) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.770870017559092) q[0];
rz(1.3250415734062564) q[0];
ry(1.6483883842060516) q[1];
rz(2.367260015742553) q[1];
ry(1.6901843700378842) q[2];
rz(2.6841374484931615) q[2];
ry(-2.1825913037228597) q[3];
rz(-2.222294761759975) q[3];
ry(-0.5027272256437674) q[4];
rz(1.4630504114578295) q[4];
ry(-1.9066148223684953) q[5];
rz(2.647013570578524) q[5];
ry(1.584968612383043) q[6];
rz(-0.018157390504627283) q[6];
ry(-0.0014769645644587825) q[7];
rz(1.7486480280629098) q[7];
ry(-3.1379804006908407) q[8];
rz(-3.056016104067868) q[8];
ry(3.137279051843486) q[9];
rz(-0.8213896612955454) q[9];
ry(-0.8849263683594843) q[10];
rz(-1.4017427552047028) q[10];
ry(3.076455730859887) q[11];
rz(1.922248400486307) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.41758101908358) q[0];
rz(-0.45776489500679657) q[0];
ry(0.34755569498121486) q[1];
rz(-1.7255288453272162) q[1];
ry(1.3014528042288824) q[2];
rz(1.497683561037379) q[2];
ry(0.022728839082335025) q[3];
rz(-1.4955758589864034) q[3];
ry(3.138058060289901) q[4];
rz(0.1490241311294387) q[4];
ry(1.5899262393545008) q[5];
rz(1.5737315064006978) q[5];
ry(-1.6603908807814793) q[6];
rz(-0.6884370049252421) q[6];
ry(2.2854818812365156) q[7];
rz(3.1081086735227683) q[7];
ry(0.318136728382699) q[8];
rz(-2.1050746197942685) q[8];
ry(0.2715422343511755) q[9];
rz(0.1102307998039107) q[9];
ry(-0.03767920323993743) q[10];
rz(-1.4902971288790594) q[10];
ry(-2.633925885745635) q[11];
rz(-0.8751761581807255) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.0400685052837542) q[0];
rz(1.3662704985411027) q[0];
ry(-0.14006246038018552) q[1];
rz(-3.1028018156750785) q[1];
ry(1.429493965150939) q[2];
rz(-2.5198718586375723) q[2];
ry(1.2414249148365712) q[3];
rz(2.3112497760214854) q[3];
ry(0.0026310168304649584) q[4];
rz(-2.654588507548438) q[4];
ry(-1.5667243652417646) q[5];
rz(1.450838069247226) q[5];
ry(-0.0010801497912158453) q[6];
rz(2.279944001590282) q[6];
ry(-1.6644202851854306) q[7];
rz(1.1920790717445486) q[7];
ry(-0.05391059952166085) q[8];
rz(-0.14931457727866626) q[8];
ry(-1.5305469519681667) q[9];
rz(-1.7760927783933933) q[9];
ry(-0.8122893991795589) q[10];
rz(-2.6493319111306666) q[10];
ry(-1.5077060330213023) q[11];
rz(-1.2549385174452914) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.6817872747803513) q[0];
rz(2.6534461615751432) q[0];
ry(-1.4780828740896823) q[1];
rz(1.9149992943739305) q[1];
ry(0.7468791389508702) q[2];
rz(-1.0514386774513038) q[2];
ry(0.11564473344822085) q[3];
rz(-1.1712305381230355) q[3];
ry(2.4490789883841395) q[4];
rz(3.061729669259895) q[4];
ry(-1.3684261804108986) q[5];
rz(2.353753736479534) q[5];
ry(-0.2701899569678501) q[6];
rz(1.281343791206968) q[6];
ry(0.7461949936588379) q[7];
rz(-2.7039179953048222) q[7];
ry(0.5741984068169604) q[8];
rz(0.6126445650487123) q[8];
ry(-0.9258025751754827) q[9];
rz(2.8708640938225387) q[9];
ry(1.7093637671670223) q[10];
rz(-1.9452770199638085) q[10];
ry(-0.9849897947080402) q[11];
rz(2.756620748761719) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.3658726265707326) q[0];
rz(-1.004093978973799) q[0];
ry(2.753780623986013) q[1];
rz(-1.983632852023658) q[1];
ry(-0.22376762596702893) q[2];
rz(0.42003336069090486) q[2];
ry(-3.134862545792783) q[3];
rz(-0.5107934137913137) q[3];
ry(3.139418870296732) q[4];
rz(3.0735069781799944) q[4];
ry(-3.141041433521356) q[5];
rz(-0.2947383161176528) q[5];
ry(-3.137424406214999) q[6];
rz(2.8647541575237594) q[6];
ry(1.5722880564696542) q[7];
rz(1.4632582961092222) q[7];
ry(0.06177015113402451) q[8];
rz(2.276244880166453) q[8];
ry(-0.5581705638784265) q[9];
rz(3.1041161126216226) q[9];
ry(-0.011595338582568893) q[10];
rz(-3.022107135067149) q[10];
ry(1.0852765719276736) q[11];
rz(0.9994284266049914) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.534261735577525) q[0];
rz(1.802042387153408) q[0];
ry(-3.112951917295529) q[1];
rz(2.67318382382232) q[1];
ry(-2.032425184633725) q[2];
rz(-0.4893850645497837) q[2];
ry(-0.09006591484888206) q[3];
rz(1.0826703961459767) q[3];
ry(0.6842621904742715) q[4];
rz(1.0535907230863444) q[4];
ry(-2.955083310061674) q[5];
rz(2.126121250096543) q[5];
ry(2.838407244878344) q[6];
rz(-0.014548650820295975) q[6];
ry(1.4876567078469487) q[7];
rz(-2.010773790466806) q[7];
ry(-0.1104994379176695) q[8];
rz(0.9707395016560042) q[8];
ry(0.9824147341446404) q[9];
rz(-2.296384202963529) q[9];
ry(3.1260371259766964) q[10];
rz(1.3263415038468485) q[10];
ry(-3.1345420405519553) q[11];
rz(0.9931179347985168) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.17292573796988941) q[0];
rz(2.522793882855731) q[0];
ry(1.0633813225839397) q[1];
rz(-0.6867206608424088) q[1];
ry(2.751146643836776) q[2];
rz(0.43872680999338487) q[2];
ry(0.04672740629453432) q[3];
rz(2.3007514679150107) q[3];
ry(-0.001129619955925641) q[4];
rz(1.3536492451815647) q[4];
ry(0.02701904365298457) q[5];
rz(-2.110688890383093) q[5];
ry(-1.5679842986683266) q[6];
rz(0.4032169141636756) q[6];
ry(2.4300136001925745) q[7];
rz(-1.415876889914305) q[7];
ry(-0.12505720496972217) q[8];
rz(2.624751480863765) q[8];
ry(-0.5661743427755503) q[9];
rz(-2.1386010505930684) q[9];
ry(0.4168771457529967) q[10];
rz(-3.1301453866977935) q[10];
ry(-2.0295904903285007) q[11];
rz(-1.9765951571576261) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.1779815428658522) q[0];
rz(-2.1606182198961754) q[0];
ry(-0.8243729696358226) q[1];
rz(-3.0901320913215993) q[1];
ry(-1.372324457690841) q[2];
rz(1.8835170946698514) q[2];
ry(-3.120729138838402) q[3];
rz(0.7334012238191844) q[3];
ry(-1.4795778767002725) q[4];
rz(2.1527180740489444) q[4];
ry(1.5636492180970833) q[5];
rz(-1.568616398428974) q[5];
ry(-3.015886034116982) q[6];
rz(0.39843229331878494) q[6];
ry(0.012094311223838689) q[7];
rz(1.8328315662272223) q[7];
ry(-3.005664317524908) q[8];
rz(-2.2218771629698466) q[8];
ry(3.082549329107761) q[9];
rz(-1.7373229389952574) q[9];
ry(-1.6779669907315915) q[10];
rz(-0.6074681462715015) q[10];
ry(0.18481695937059683) q[11];
rz(0.4248715860079834) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.6569852119647109) q[0];
rz(3.136244319432034) q[0];
ry(2.631631807838196) q[1];
rz(3.018838364449311) q[1];
ry(0.05105504137751549) q[2];
rz(1.326652072173969) q[2];
ry(-0.4828192246665891) q[3];
rz(0.4199153041543591) q[3];
ry(0.7571712925297687) q[4];
rz(-0.5360527773382772) q[4];
ry(-1.380751922533985) q[5];
rz(-1.0049141320346513) q[5];
ry(1.66050499477089) q[6];
rz(-1.577959492208426) q[6];
ry(-0.5611797199529904) q[7];
rz(-1.805190593130315) q[7];
ry(0.09949770029025752) q[8];
rz(2.526730401894998) q[8];
ry(0.7582484680305486) q[9];
rz(1.4710693199140596) q[9];
ry(1.2462852291960642) q[10];
rz(-1.4534575888081063) q[10];
ry(2.8609023329293755) q[11];
rz(-2.5594595838372665) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.925320525331704) q[0];
rz(1.3325138887687054) q[0];
ry(0.7829385255493478) q[1];
rz(1.0373660353181053) q[1];
ry(1.55196985809794) q[2];
rz(0.5824203071711871) q[2];
ry(-0.0025717188721939704) q[3];
rz(-3.099860461166595) q[3];
ry(-0.0013259396166285242) q[4];
rz(0.4751105202229623) q[4];
ry(-3.138885673448395) q[5];
rz(-1.0053133463953055) q[5];
ry(-1.5719869643291249) q[6];
rz(0.19153330216082154) q[6];
ry(-1.5406862931458087) q[7];
rz(-2.9602813176843012) q[7];
ry(-2.9750132987586175) q[8];
rz(-3.0251395665631264) q[8];
ry(0.010056155506827511) q[9];
rz(0.9523793448324394) q[9];
ry(-3.14102732860155) q[10];
rz(-0.11872355951851976) q[10];
ry(2.4818099412435193) q[11];
rz(-0.7025408993476798) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.49230459499864) q[0];
rz(0.4925487969811655) q[0];
ry(1.7648857720366637) q[1];
rz(0.9166008791587165) q[1];
ry(-1.6705669230964075) q[2];
rz(-2.7595125829247293) q[2];
ry(3.1336938499535725) q[3];
rz(-1.7471482298762204) q[3];
ry(-2.4083874340198945) q[4];
rz(2.1090110637553776) q[4];
ry(-1.5810150869925472) q[5];
rz(0.0008856118747392902) q[5];
ry(-3.1394601826257995) q[6];
rz(-2.0626663032876893) q[6];
ry(3.138564611116597) q[7];
rz(-1.4368923624656817) q[7];
ry(1.5538397384678884) q[8];
rz(-0.0019089190978762716) q[8];
ry(3.0136098978797405) q[9];
rz(-0.7410722242180912) q[9];
ry(-3.112387359071529) q[10];
rz(-0.557054975862126) q[10];
ry(1.0206460014705734) q[11];
rz(-0.47581193487459444) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9698796212250776) q[0];
rz(1.9555920038093761) q[0];
ry(3.1388234790721707) q[1];
rz(-2.2585897549375002) q[1];
ry(0.004112179598346952) q[2];
rz(2.615647341867828) q[2];
ry(-3.1380422118425146) q[3];
rz(-0.6421996594251445) q[3];
ry(-1.5702801625057248) q[4];
rz(1.7196329932036711) q[4];
ry(-1.4686596888333447) q[5];
rz(3.1247964488577775) q[5];
ry(2.876820459822813) q[6];
rz(-2.229470564377177) q[6];
ry(-0.0014205876372148651) q[7];
rz(-1.5351088504519446) q[7];
ry(1.7427514929840893) q[8];
rz(3.130686278863917) q[8];
ry(-0.6919983202998481) q[9];
rz(-3.140145167141207) q[9];
ry(0.0032205653118971764) q[10];
rz(2.138112261572643) q[10];
ry(-1.9461749630314187) q[11];
rz(-1.0286471444170981) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.09281527353883644) q[0];
rz(-1.5777655309110357) q[0];
ry(0.3698532482219905) q[1];
rz(1.6072326419883893) q[1];
ry(-1.2683160739343382) q[2];
rz(-2.465332152752119) q[2];
ry(-3.0548017509560363) q[3];
rz(0.8557382080895521) q[3];
ry(-2.467543708106071) q[4];
rz(1.8590144963701383) q[4];
ry(1.4400520901844454) q[5];
rz(0.8884383505851542) q[5];
ry(3.1247848120636483) q[6];
rz(0.029938566991859) q[6];
ry(0.7697555190371181) q[7];
rz(0.005935653290806878) q[7];
ry(-1.7924129496503811) q[8];
rz(0.9070691190059417) q[8];
ry(-1.5493183778919386) q[9];
rz(-3.1400970073660095) q[9];
ry(-1.1233261255731963) q[10];
rz(1.84520301013409) q[10];
ry(0.7192703297613584) q[11];
rz(-1.9299758877708573) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.4768653502584104) q[0];
rz(0.5164793695408617) q[0];
ry(1.3694881600979958) q[1];
rz(-0.016974190657950056) q[1];
ry(-1.6604134653636426) q[2];
rz(2.2289174802016802) q[2];
ry(-1.7554702692379465e-05) q[3];
rz(0.29760481863639565) q[3];
ry(3.1401109035737855) q[4];
rz(2.6922415971220595) q[4];
ry(-3.1383254569574834) q[5];
rz(-2.607843112403955) q[5];
ry(-2.9795083385821077) q[6];
rz(-1.8929238268031536) q[6];
ry(2.447035835610886) q[7];
rz(-0.0028179748166632383) q[7];
ry(0.010178227044032174) q[8];
rz(2.2053863970725764) q[8];
ry(-0.7943341382381275) q[9];
rz(1.1784362646053304) q[9];
ry(-3.0921303761083783) q[10];
rz(-3.050383974651047) q[10];
ry(-2.2504206091216226) q[11];
rz(0.7721068120922742) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.3648198929613908) q[0];
rz(1.8442348643764486) q[0];
ry(-1.5699190334273392) q[1];
rz(0.2764778091883641) q[1];
ry(0.10272023553918237) q[2];
rz(-1.3465803061123918) q[2];
ry(-0.02258033731735358) q[3];
rz(-2.993052777834806) q[3];
ry(-2.113495829392444) q[4];
rz(2.3887309439946685) q[4];
ry(-0.03190323137423122) q[5];
rz(-2.764272681558565) q[5];
ry(3.250600502191503e-05) q[6];
rz(2.9277173478391285) q[6];
ry(2.335372276944973) q[7];
rz(3.1392417407273125) q[7];
ry(0.372923269696229) q[8];
rz(0.031012556055941328) q[8];
ry(-5.759318529285906e-05) q[9];
rz(1.9615037613128632) q[9];
ry(-0.003732457217711693) q[10];
rz(0.8240334360019412) q[10];
ry(-1.266890992591386) q[11];
rz(3.0674262694072416) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.4098913772953043) q[0];
rz(2.964719677768933) q[0];
ry(0.0441908539405178) q[1];
rz(2.041209777584549) q[1];
ry(-0.02455580595705066) q[2];
rz(1.4787437211410455) q[2];
ry(3.128198739893542) q[3];
rz(2.871775466753491) q[3];
ry(-3.116875012633759) q[4];
rz(-3.0127273581345957) q[4];
ry(-0.07198598218791963) q[5];
rz(3.117203222393118) q[5];
ry(-3.1414356636584917) q[6];
rz(1.030752995260808) q[6];
ry(2.444737193786593) q[7];
rz(0.0030351582341902094) q[7];
ry(-1.5766463346788815) q[8];
rz(-1.3744776498364673) q[8];
ry(2.389232254716792) q[9];
rz(3.135039678439641) q[9];
ry(-3.109232563567475) q[10];
rz(-0.935888148993274) q[10];
ry(2.297876578032722) q[11];
rz(-0.7748275748076979) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.04072148418613169) q[0];
rz(2.09880342186379) q[0];
ry(0.0034910011198290296) q[1];
rz(-2.0247838621947647) q[1];
ry(-3.0488277582707157) q[2];
rz(0.7876435108472353) q[2];
ry(0.44022906724331484) q[3];
rz(-3.1415495089373855) q[3];
ry(2.53449670745001) q[4];
rz(2.488225144234479) q[4];
ry(2.092800941821549) q[5];
rz(1.6032009285913704) q[5];
ry(3.0267554285214326) q[6];
rz(0.02044080985573648) q[6];
ry(-2.41799556141995) q[7];
rz(0.07529941025657817) q[7];
ry(-3.1402617144985006) q[8];
rz(-2.937606394004151) q[8];
ry(1.5733947533130719) q[9];
rz(-3.1410210479352196) q[9];
ry(-1.6102546371827495) q[10];
rz(-0.0048649211864970855) q[10];
ry(-2.674585501115136) q[11];
rz(3.0056815938887778) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.4291801232892647) q[0];
rz(-1.405515448697044) q[0];
ry(-0.6672118441410699) q[1];
rz(-1.8089363839782244) q[1];
ry(1.5825740070380816) q[2];
rz(-0.3136495521220706) q[2];
ry(-0.5097095866797908) q[3];
rz(-0.11660731541573077) q[3];
ry(3.140552019453611) q[4];
rz(-0.7106115281624437) q[4];
ry(-3.141086701794594) q[5];
rz(1.6043271538541113) q[5];
ry(0.14232089566310208) q[6];
rz(0.43670531865546425) q[6];
ry(3.1364915785019685) q[7];
rz(1.6437972315914346) q[7];
ry(-1.5693806002786843) q[8];
rz(2.3315181008734456) q[8];
ry(1.6141925345869872) q[9];
rz(-0.0008936321639697118) q[9];
ry(1.5709118355610958) q[10];
rz(-1.5719375880244302) q[10];
ry(0.3320535820038165) q[11];
rz(0.6805595748042459) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.1674586930228328) q[0];
rz(3.1350269830212234) q[0];
ry(1.5699929914161563) q[1];
rz(0.6191044346775998) q[1];
ry(0.017191063120312933) q[2];
rz(0.3830296593557088) q[2];
ry(-3.1355957988771577) q[3];
rz(1.2573039417185172) q[3];
ry(-1.720003750059865) q[4];
rz(2.791220525016686) q[4];
ry(-2.512033172141071) q[5];
rz(-0.004773910361907534) q[5];
ry(-3.1394713229965054) q[6];
rz(0.4731567935641303) q[6];
ry(0.0689716320031799) q[7];
rz(0.9295446371019978) q[7];
ry(5.831210817675015e-05) q[8];
rz(2.3052954621871637) q[8];
ry(-1.571025717608585) q[9];
rz(-1.57598676830201) q[9];
ry(-1.5707298328853407) q[10];
rz(-3.101031878769455) q[10];
ry(1.5693910690318216) q[11];
rz(-1.258047738163964) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5716346621801967) q[0];
rz(3.13562151706448) q[0];
ry(-0.08030199733836947) q[1];
rz(0.9536758281230738) q[1];
ry(-0.017639638861586704) q[2];
rz(2.6812846191011643) q[2];
ry(-0.0757170672896447) q[3];
rz(0.060369202966109384) q[3];
ry(2.0659430602289888) q[4];
rz(-1.9204040335772974) q[4];
ry(1.5686413462249411) q[5];
rz(-3.036457911162452) q[5];
ry(0.6081279899947996) q[6];
rz(-3.1352280185821226) q[6];
ry(-3.1401621702828133) q[7];
rz(1.6641868973616154) q[7];
ry(-1.583087599609085) q[8];
rz(2.2800811213796246) q[8];
ry(1.5709448838528082) q[9];
rz(3.141064412782089) q[9];
ry(3.064443125412441) q[10];
rz(1.6119683150988227) q[10];
ry(-3.137734758077706) q[11];
rz(0.313366918718721) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.08701545889606699) q[0];
rz(2.0151348199899255) q[0];
ry(-1.570897905389617) q[1];
rz(0.043431756405544426) q[1];
ry(-3.1251545503052336) q[2];
rz(-0.38929122850704) q[2];
ry(-0.00039289480469317795) q[3];
rz(0.13378678656662754) q[3];
ry(-3.1401202196927054) q[4];
rz(2.964850817387414) q[4];
ry(0.002685841422306042) q[5];
rz(-1.6767527201838792) q[5];
ry(1.5678419196026974) q[6];
rz(1.5704460088503964) q[6];
ry(-0.0019756849517511554) q[7];
rz(-0.7378556848876014) q[7];
ry(-0.0004261572917245715) q[8];
rz(3.0819137006880477) q[8];
ry(1.5704796977234035) q[9];
rz(-1.5313880840675098) q[9];
ry(1.5684131681681899) q[10];
rz(2.9488932958242002) q[10];
ry(-1.5221401977362925) q[11];
rz(-1.5699206266137145) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.138838152981565) q[0];
rz(-2.116820835857709) q[0];
ry(-0.04712679140234677) q[1];
rz(-2.6989118716580482) q[1];
ry(1.5705582312641901) q[2];
rz(0.5954807661015175) q[2];
ry(-1.4957377210901486) q[3];
rz(-2.731353634625147) q[3];
ry(1.8781503466627743) q[4];
rz(-2.0069346344995376) q[4];
ry(-1.5702215134106998) q[5];
rz(1.7998886948947002) q[5];
ry(1.5708512052243917) q[6];
rz(-0.4702539247902786) q[6];
ry(1.571015023412831) q[7];
rz(0.12727395468933667) q[7];
ry(0.00021405840799133152) q[8];
rz(-0.042335705941692225) q[8];
ry(3.140958242074114) q[9];
rz(1.724591612633367) q[9];
ry(-3.1414802687349153) q[10];
rz(-2.7193573988040556) q[10];
ry(1.5705383445752776) q[11];
rz(-3.0116633896745566) q[11];