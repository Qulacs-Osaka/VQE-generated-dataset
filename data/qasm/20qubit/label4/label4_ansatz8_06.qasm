OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.4430653746676887) q[0];
ry(-2.802444395342356) q[1];
cx q[0],q[1];
ry(-1.4506255716863459) q[0];
ry(-0.845867505254402) q[1];
cx q[0],q[1];
ry(2.7614684870165873) q[2];
ry(-2.054964174363284) q[3];
cx q[2],q[3];
ry(1.0888932253288681) q[2];
ry(-0.593811264554863) q[3];
cx q[2],q[3];
ry(-2.9189787687577575) q[4];
ry(-2.0496663618230935) q[5];
cx q[4],q[5];
ry(0.44741390441303164) q[4];
ry(2.7949328104789) q[5];
cx q[4],q[5];
ry(-0.5026333448486939) q[6];
ry(0.30204171789281836) q[7];
cx q[6],q[7];
ry(2.157220160924383) q[6];
ry(-1.4278132459200812) q[7];
cx q[6],q[7];
ry(1.3178989618826442) q[8];
ry(0.0712480017934492) q[9];
cx q[8],q[9];
ry(-2.9626340137286777) q[8];
ry(-2.9667183479650925) q[9];
cx q[8],q[9];
ry(-0.3102399806979801) q[10];
ry(-0.5381154198595262) q[11];
cx q[10],q[11];
ry(-1.9612699864277807) q[10];
ry(2.1257342092571507) q[11];
cx q[10],q[11];
ry(0.008621076139346151) q[12];
ry(-2.7685956795241484) q[13];
cx q[12],q[13];
ry(2.6498051995950584) q[12];
ry(-1.6134238784711152) q[13];
cx q[12],q[13];
ry(-1.7208678285595918) q[14];
ry(-2.573351835639166) q[15];
cx q[14],q[15];
ry(0.7573112284267962) q[14];
ry(0.3990644437385669) q[15];
cx q[14],q[15];
ry(-2.832443920605883) q[16];
ry(-2.3449849771800353) q[17];
cx q[16],q[17];
ry(2.3966552709744917) q[16];
ry(-0.4977719991569228) q[17];
cx q[16],q[17];
ry(1.544806301969679) q[18];
ry(-2.106349999053762) q[19];
cx q[18],q[19];
ry(-1.6144247263348825) q[18];
ry(-2.348564751878754) q[19];
cx q[18],q[19];
ry(-3.105799435398754) q[0];
ry(-0.30073130864876124) q[2];
cx q[0],q[2];
ry(-0.4195521730409183) q[0];
ry(0.690943566961377) q[2];
cx q[0],q[2];
ry(2.4758131759947184) q[2];
ry(-0.42351557473905693) q[4];
cx q[2],q[4];
ry(-1.852275183082443) q[2];
ry(1.1247985150217845) q[4];
cx q[2],q[4];
ry(0.9771048928193604) q[4];
ry(1.7577499841766793) q[6];
cx q[4],q[6];
ry(-2.49839426585785) q[4];
ry(1.0266967768179909) q[6];
cx q[4],q[6];
ry(1.354718592324815) q[6];
ry(2.505947793612932) q[8];
cx q[6],q[8];
ry(-0.0054100015905450115) q[6];
ry(-0.22865082586645938) q[8];
cx q[6],q[8];
ry(-0.7734001213923385) q[8];
ry(-2.719193891018248) q[10];
cx q[8],q[10];
ry(1.1773730916034788) q[8];
ry(-2.442006069424699) q[10];
cx q[8],q[10];
ry(2.1606435137726296) q[10];
ry(-2.9813735181638585) q[12];
cx q[10],q[12];
ry(-3.011133357848589) q[10];
ry(-2.516258179558962) q[12];
cx q[10],q[12];
ry(0.6043212203938201) q[12];
ry(-3.0548553058590926) q[14];
cx q[12],q[14];
ry(3.114222134085967) q[12];
ry(3.0510185475285803) q[14];
cx q[12],q[14];
ry(0.3837645321298595) q[14];
ry(1.4281080218895275) q[16];
cx q[14],q[16];
ry(0.07723527604627911) q[14];
ry(-3.061419152749721) q[16];
cx q[14],q[16];
ry(0.6326873419787992) q[16];
ry(-0.050902218317384916) q[18];
cx q[16],q[18];
ry(3.125036345129198) q[16];
ry(0.5915569461681979) q[18];
cx q[16],q[18];
ry(-2.292345315198865) q[1];
ry(-3.0385010088645554) q[3];
cx q[1],q[3];
ry(2.7541789255915443) q[1];
ry(-1.2085349517932904) q[3];
cx q[1],q[3];
ry(2.984317000186485) q[3];
ry(-0.5728285514467553) q[5];
cx q[3],q[5];
ry(-2.0799476115775084) q[3];
ry(-0.6761790271066914) q[5];
cx q[3],q[5];
ry(-0.8227384903004102) q[5];
ry(-0.5366326495014955) q[7];
cx q[5],q[7];
ry(0.2650877673770795) q[5];
ry(-0.9344234051705752) q[7];
cx q[5],q[7];
ry(0.7411282267201218) q[7];
ry(-1.4591644342640562) q[9];
cx q[7],q[9];
ry(0.93658273774051) q[7];
ry(-3.139954185356737) q[9];
cx q[7],q[9];
ry(-2.6601928650283058) q[9];
ry(3.0690736652942383) q[11];
cx q[9],q[11];
ry(-0.23298958620566257) q[9];
ry(-0.05572260331430545) q[11];
cx q[9],q[11];
ry(2.106894185696239) q[11];
ry(-1.3672968223212587) q[13];
cx q[11],q[13];
ry(1.9192051148749627) q[11];
ry(1.7359635482827187) q[13];
cx q[11],q[13];
ry(-1.2341429331129312) q[13];
ry(0.12659899329566038) q[15];
cx q[13],q[15];
ry(-0.011361955844236602) q[13];
ry(2.5355161904838583) q[15];
cx q[13],q[15];
ry(-2.3002398311290095) q[15];
ry(-0.30562279399295783) q[17];
cx q[15],q[17];
ry(1.6533333791162557) q[15];
ry(-0.015881561130793916) q[17];
cx q[15],q[17];
ry(2.283027086287853) q[17];
ry(1.9095013319359184) q[19];
cx q[17],q[19];
ry(-1.6359461296598043) q[17];
ry(-0.7790677229064986) q[19];
cx q[17],q[19];
ry(2.956357990873641) q[0];
ry(-1.8469250528329468) q[1];
cx q[0],q[1];
ry(2.0085947880383603) q[0];
ry(1.2545711884597455) q[1];
cx q[0],q[1];
ry(2.247921653879243) q[2];
ry(-1.1574389532988358) q[3];
cx q[2],q[3];
ry(2.4876450247311297) q[2];
ry(0.5114255235952455) q[3];
cx q[2],q[3];
ry(0.44440620500769334) q[4];
ry(2.877341130837579) q[5];
cx q[4],q[5];
ry(2.2947710641225942) q[4];
ry(-3.1362153174339125) q[5];
cx q[4],q[5];
ry(-1.6214476918981529) q[6];
ry(-1.1051553869309227) q[7];
cx q[6],q[7];
ry(1.6742108911403282) q[6];
ry(-1.5809709101649005) q[7];
cx q[6],q[7];
ry(-3.0482243568449268) q[8];
ry(-0.8873262308347982) q[9];
cx q[8],q[9];
ry(-1.5815874394250242) q[8];
ry(-0.03930556791008222) q[9];
cx q[8],q[9];
ry(1.9402966157271644) q[10];
ry(1.2873061042111114) q[11];
cx q[10],q[11];
ry(1.698921032264284) q[10];
ry(1.6920451041425544) q[11];
cx q[10],q[11];
ry(1.9240084545228684) q[12];
ry(1.7932867247066895) q[13];
cx q[12],q[13];
ry(-1.8478092774877624) q[12];
ry(0.08899710446090747) q[13];
cx q[12],q[13];
ry(-0.282700923672801) q[14];
ry(-1.8871840243554727) q[15];
cx q[14],q[15];
ry(-1.7144063992745568) q[14];
ry(2.3502650085608874) q[15];
cx q[14],q[15];
ry(-3.104549288456156) q[16];
ry(-0.9053549914174405) q[17];
cx q[16],q[17];
ry(3.1364927805747977) q[16];
ry(0.3260851345782436) q[17];
cx q[16],q[17];
ry(-2.2920549533693055) q[18];
ry(0.40956198482503986) q[19];
cx q[18],q[19];
ry(1.617904950239493) q[18];
ry(-1.2211771625939907) q[19];
cx q[18],q[19];
ry(0.5043837409478904) q[0];
ry(-0.3807600294213191) q[2];
cx q[0],q[2];
ry(-2.229877630816112) q[0];
ry(2.790683620416705) q[2];
cx q[0],q[2];
ry(1.6372206637762263) q[2];
ry(1.5934219076298524) q[4];
cx q[2],q[4];
ry(-3.119500912704698) q[2];
ry(1.8603158546284124) q[4];
cx q[2],q[4];
ry(1.136614196927128) q[4];
ry(2.718375533938721) q[6];
cx q[4],q[6];
ry(0.6754571478533924) q[4];
ry(-2.962255647217274) q[6];
cx q[4],q[6];
ry(1.5695460920730602) q[6];
ry(-2.2997538546957332) q[8];
cx q[6],q[8];
ry(-0.00998278597976121) q[6];
ry(-3.126706431848484) q[8];
cx q[6],q[8];
ry(-1.3246990328340784) q[8];
ry(1.2754286090830806) q[10];
cx q[8],q[10];
ry(0.3297058170878371) q[8];
ry(2.5322842668281593) q[10];
cx q[8],q[10];
ry(-1.9048836685269863) q[10];
ry(0.06542301954333225) q[12];
cx q[10],q[12];
ry(-3.1244478375799565) q[10];
ry(-0.05860365222931829) q[12];
cx q[10],q[12];
ry(0.2007330756803173) q[12];
ry(1.8236317426982662) q[14];
cx q[12],q[14];
ry(-1.3311496512069707) q[12];
ry(-2.8725720199602516) q[14];
cx q[12],q[14];
ry(-2.008552286973204) q[14];
ry(-1.5970543812681752) q[16];
cx q[14],q[16];
ry(2.4001346243791684) q[14];
ry(-3.1237338726961292) q[16];
cx q[14],q[16];
ry(-2.4611775828035882) q[16];
ry(3.0094745480543503) q[18];
cx q[16],q[18];
ry(-2.4368590208276775) q[16];
ry(0.3185795503419433) q[18];
cx q[16],q[18];
ry(-0.9796910534884029) q[1];
ry(1.2218084159260556) q[3];
cx q[1],q[3];
ry(-1.0852071043475415) q[1];
ry(2.603996948265392) q[3];
cx q[1],q[3];
ry(-0.646077954132384) q[3];
ry(2.5730413507756458) q[5];
cx q[3],q[5];
ry(-3.029299043007449) q[3];
ry(-1.413854583993674) q[5];
cx q[3],q[5];
ry(1.5247743822873019) q[5];
ry(-1.5998680681170272) q[7];
cx q[5],q[7];
ry(2.527660362095307) q[5];
ry(0.021982557602722608) q[7];
cx q[5],q[7];
ry(1.544068244513902) q[7];
ry(0.9698796368395026) q[9];
cx q[7],q[9];
ry(0.00029706584242161) q[7];
ry(3.114103439404588) q[9];
cx q[7],q[9];
ry(2.720074715156647) q[9];
ry(2.3116156921346738) q[11];
cx q[9],q[11];
ry(-3.1091683095542897) q[9];
ry(3.1071491117009638) q[11];
cx q[9],q[11];
ry(0.9665514625319691) q[11];
ry(-1.5087659885075195) q[13];
cx q[11],q[13];
ry(1.587385448704632) q[11];
ry(-1.5686563284630979) q[13];
cx q[11],q[13];
ry(-1.8286825487143359) q[13];
ry(0.6150051010573659) q[15];
cx q[13],q[15];
ry(-0.4371955023883371) q[13];
ry(-1.5764773579289824) q[15];
cx q[13],q[15];
ry(1.078717263564089) q[15];
ry(-1.4917852108373628) q[17];
cx q[15],q[17];
ry(3.1408800738268328) q[15];
ry(0.0047768898662137625) q[17];
cx q[15],q[17];
ry(-2.9098292864616586) q[17];
ry(3.044442689699405) q[19];
cx q[17],q[19];
ry(1.558901576246286) q[17];
ry(-2.1816912608035484) q[19];
cx q[17],q[19];
ry(-0.6536277647575387) q[0];
ry(-0.7241323000742371) q[1];
cx q[0],q[1];
ry(-2.113013663523624) q[0];
ry(-3.040565240980125) q[1];
cx q[0],q[1];
ry(-2.7076388494473367) q[2];
ry(-0.9998375440558033) q[3];
cx q[2],q[3];
ry(-0.7387997731267727) q[2];
ry(0.19470689935216345) q[3];
cx q[2],q[3];
ry(-2.9607112747787143) q[4];
ry(2.9951799577902234) q[5];
cx q[4],q[5];
ry(0.0198494005343173) q[4];
ry(-2.7966078580310243) q[5];
cx q[4],q[5];
ry(-0.1195292754450703) q[6];
ry(-1.0066646158439563) q[7];
cx q[6],q[7];
ry(-0.9761478591665861) q[6];
ry(3.1020800218404005) q[7];
cx q[6],q[7];
ry(1.6415454410360835) q[8];
ry(0.6756929048307301) q[9];
cx q[8],q[9];
ry(2.0839056261823607) q[8];
ry(0.02055432304026785) q[9];
cx q[8],q[9];
ry(-2.061281763885017) q[10];
ry(-0.07236526497608153) q[11];
cx q[10],q[11];
ry(1.4922483504929196) q[10];
ry(-2.7799751555895327) q[11];
cx q[10],q[11];
ry(-1.604888204241395) q[12];
ry(-0.9869454850306508) q[13];
cx q[12],q[13];
ry(-1.9304217523383898) q[12];
ry(-2.777777353682702) q[13];
cx q[12],q[13];
ry(-2.691503247304772) q[14];
ry(1.1320766843523675) q[15];
cx q[14],q[15];
ry(2.672907966989185) q[14];
ry(-0.4805345329275621) q[15];
cx q[14],q[15];
ry(2.9139214868951453) q[16];
ry(1.913868282395825) q[17];
cx q[16],q[17];
ry(1.4456407588801568) q[16];
ry(1.4134948379436811) q[17];
cx q[16],q[17];
ry(3.0108540010865723) q[18];
ry(1.9253928542126753) q[19];
cx q[18],q[19];
ry(1.8904577663499342) q[18];
ry(2.4641844797780683) q[19];
cx q[18],q[19];
ry(0.8032752902760034) q[0];
ry(-0.6782376724763465) q[2];
cx q[0],q[2];
ry(-1.2166508656197945) q[0];
ry(0.26838763486936235) q[2];
cx q[0],q[2];
ry(0.8072798168893821) q[2];
ry(-1.1558806405980153) q[4];
cx q[2],q[4];
ry(-2.742118458531295) q[2];
ry(1.8077042382085244) q[4];
cx q[2],q[4];
ry(1.2111511262427606) q[4];
ry(2.1230307238724047) q[6];
cx q[4],q[6];
ry(0.14536999691418315) q[4];
ry(-0.3058457356542785) q[6];
cx q[4],q[6];
ry(2.945285928062779) q[6];
ry(-2.6123429330676156) q[8];
cx q[6],q[8];
ry(0.0075682299693976135) q[6];
ry(3.138135253966537) q[8];
cx q[6],q[8];
ry(-1.9021851803992877) q[8];
ry(0.7046508368866518) q[10];
cx q[8],q[10];
ry(2.7364247310515917) q[8];
ry(1.4512486856186781) q[10];
cx q[8],q[10];
ry(-0.4683700036456531) q[10];
ry(1.5155272660483476) q[12];
cx q[10],q[12];
ry(3.1401214725424893) q[10];
ry(0.011306915085580016) q[12];
cx q[10],q[12];
ry(-2.6187521872127038) q[12];
ry(0.8897763858998324) q[14];
cx q[12],q[14];
ry(2.707469428231556) q[12];
ry(-0.0665828776868107) q[14];
cx q[12],q[14];
ry(1.6815483839185983) q[14];
ry(0.8866970013714787) q[16];
cx q[14],q[16];
ry(3.1350592041930487) q[14];
ry(-3.133042995477339) q[16];
cx q[14],q[16];
ry(-1.3488508644844615) q[16];
ry(2.457173965152244) q[18];
cx q[16],q[18];
ry(-0.2120208414987873) q[16];
ry(2.83340477559709) q[18];
cx q[16],q[18];
ry(0.34374217867221457) q[1];
ry(0.7519853233634781) q[3];
cx q[1],q[3];
ry(0.21463392188595698) q[1];
ry(1.7027412843832128) q[3];
cx q[1],q[3];
ry(-0.9765996918263242) q[3];
ry(2.8707634123011623) q[5];
cx q[3],q[5];
ry(0.3653803450835955) q[3];
ry(-2.2222375891155) q[5];
cx q[3],q[5];
ry(2.8046476782742893) q[5];
ry(-1.9497760679627856) q[7];
cx q[5],q[7];
ry(-3.1398877800347464) q[5];
ry(-0.06654481255033605) q[7];
cx q[5],q[7];
ry(-2.6273714355633584) q[7];
ry(-2.2903853142062722) q[9];
cx q[7],q[9];
ry(-0.0025629161215724823) q[7];
ry(0.00021453089686751525) q[9];
cx q[7],q[9];
ry(-2.9156173815849016) q[9];
ry(1.6012301544184728) q[11];
cx q[9],q[11];
ry(-0.01772120463384219) q[9];
ry(-0.03240194561535148) q[11];
cx q[9],q[11];
ry(2.643735799070681) q[11];
ry(1.8638377449468249) q[13];
cx q[11],q[13];
ry(-3.105544593800213) q[11];
ry(-3.0231515185605415) q[13];
cx q[11],q[13];
ry(0.023771756757541063) q[13];
ry(0.5294059652282233) q[15];
cx q[13],q[15];
ry(2.0936446533205455) q[13];
ry(1.5250982209867383) q[15];
cx q[13],q[15];
ry(-2.0997760077948806) q[15];
ry(-1.7827680734987146) q[17];
cx q[15],q[17];
ry(3.130834342974043) q[15];
ry(0.002162149124526127) q[17];
cx q[15],q[17];
ry(0.7671098555583882) q[17];
ry(0.3722772774520693) q[19];
cx q[17],q[19];
ry(2.9543708447838783) q[17];
ry(2.7127510027683592) q[19];
cx q[17],q[19];
ry(1.3134266561612375) q[0];
ry(-1.2871980114486608) q[1];
cx q[0],q[1];
ry(-1.8912094254402156) q[0];
ry(0.23872668501297323) q[1];
cx q[0],q[1];
ry(0.9102696103289643) q[2];
ry(-2.0261604360377126) q[3];
cx q[2],q[3];
ry(0.6258897938246137) q[2];
ry(0.33633435819520585) q[3];
cx q[2],q[3];
ry(-0.7162793597835524) q[4];
ry(1.2591987819323611) q[5];
cx q[4],q[5];
ry(2.6980613148486823) q[4];
ry(0.6765526025916698) q[5];
cx q[4],q[5];
ry(1.2368378604098051) q[6];
ry(0.4635305396296154) q[7];
cx q[6],q[7];
ry(-0.8378366439461482) q[6];
ry(0.06247959386123991) q[7];
cx q[6],q[7];
ry(1.1524118393339893) q[8];
ry(-2.112241281899463) q[9];
cx q[8],q[9];
ry(3.0113759410157166) q[8];
ry(-1.5256698056003692) q[9];
cx q[8],q[9];
ry(-1.9706518236861026) q[10];
ry(3.0431543240383245) q[11];
cx q[10],q[11];
ry(0.1370901169397011) q[10];
ry(-1.6382541875998105) q[11];
cx q[10],q[11];
ry(0.23150552005972222) q[12];
ry(-0.2622709231545999) q[13];
cx q[12],q[13];
ry(0.9895699917852899) q[12];
ry(-1.6863150969501455) q[13];
cx q[12],q[13];
ry(1.784362696795668) q[14];
ry(-2.984243472758371) q[15];
cx q[14],q[15];
ry(-2.3582870171248134) q[14];
ry(-0.6639559040130347) q[15];
cx q[14],q[15];
ry(2.929475819669225) q[16];
ry(-3.0474770891753544) q[17];
cx q[16],q[17];
ry(-1.5316589296656686) q[16];
ry(-0.5807679130705932) q[17];
cx q[16],q[17];
ry(-0.9232693519163283) q[18];
ry(-1.8692871754062965) q[19];
cx q[18],q[19];
ry(0.7209303332698349) q[18];
ry(2.7354165074663546) q[19];
cx q[18],q[19];
ry(2.627476360692035) q[0];
ry(0.7161944473284149) q[2];
cx q[0],q[2];
ry(0.4550150720779955) q[0];
ry(-2.080897920348357) q[2];
cx q[0],q[2];
ry(2.253431513019521) q[2];
ry(0.9935419600964749) q[4];
cx q[2],q[4];
ry(-2.2433699647339758) q[2];
ry(-0.35911419475114403) q[4];
cx q[2],q[4];
ry(-2.7665470866973143) q[4];
ry(-2.247200299235019) q[6];
cx q[4],q[6];
ry(3.1050660133007204) q[4];
ry(0.3194669706217397) q[6];
cx q[4],q[6];
ry(-1.8257246396098) q[6];
ry(-1.2406792890209246) q[8];
cx q[6],q[8];
ry(-3.133188031036421) q[6];
ry(-2.960834859559073) q[8];
cx q[6],q[8];
ry(-2.778122401164207) q[8];
ry(3.1173660408074033) q[10];
cx q[8],q[10];
ry(0.025617540180311593) q[8];
ry(-1.6797864516437917) q[10];
cx q[8],q[10];
ry(0.08929845224420771) q[10];
ry(0.4217637576265859) q[12];
cx q[10],q[12];
ry(3.097358194945074) q[10];
ry(0.001000367978415284) q[12];
cx q[10],q[12];
ry(-1.129589198532015) q[12];
ry(2.485988841400827) q[14];
cx q[12],q[14];
ry(3.139511155398233) q[12];
ry(2.013846598986915) q[14];
cx q[12],q[14];
ry(-3.030109546601916) q[14];
ry(1.5922400999703887) q[16];
cx q[14],q[16];
ry(3.133809056348874) q[14];
ry(3.140926794019608) q[16];
cx q[14],q[16];
ry(-1.000077790275446) q[16];
ry(-0.6900607522024126) q[18];
cx q[16],q[18];
ry(1.0676425183282623) q[16];
ry(-3.066687118119442) q[18];
cx q[16],q[18];
ry(2.455215719065207) q[1];
ry(0.4466921192931272) q[3];
cx q[1],q[3];
ry(-0.12267097054069963) q[1];
ry(2.9145404126565975) q[3];
cx q[1],q[3];
ry(-2.2306007386551325) q[3];
ry(1.977636444946806) q[5];
cx q[3],q[5];
ry(-0.1227067180478496) q[3];
ry(-0.49532730398450947) q[5];
cx q[3],q[5];
ry(-1.0647623511403586) q[5];
ry(0.6948149337299627) q[7];
cx q[5],q[7];
ry(-0.7282332470916216) q[5];
ry(3.0596360190267924) q[7];
cx q[5],q[7];
ry(-0.47250781559232197) q[7];
ry(0.8288017984180501) q[9];
cx q[7],q[9];
ry(3.140103498190055) q[7];
ry(-3.137635278440058) q[9];
cx q[7],q[9];
ry(2.157125394317097) q[9];
ry(-0.39292690869991653) q[11];
cx q[9],q[11];
ry(1.5990495059265493) q[9];
ry(-0.03391910997924441) q[11];
cx q[9],q[11];
ry(1.2665069523846453) q[11];
ry(-1.7621923648854467) q[13];
cx q[11],q[13];
ry(-0.00045283778838016393) q[11];
ry(0.03044970268476899) q[13];
cx q[11],q[13];
ry(-2.1693098182330983) q[13];
ry(-2.3254771569431654) q[15];
cx q[13],q[15];
ry(2.2547667861925635) q[13];
ry(-0.584948651680461) q[15];
cx q[13],q[15];
ry(2.113058843104751) q[15];
ry(2.525655141551359) q[17];
cx q[15],q[17];
ry(-3.1323997826793923) q[15];
ry(0.005798603780393534) q[17];
cx q[15],q[17];
ry(-0.6301346768092762) q[17];
ry(0.4742467612365424) q[19];
cx q[17],q[19];
ry(-0.15346421170363275) q[17];
ry(-2.398419930657938) q[19];
cx q[17],q[19];
ry(-2.842155689427151) q[0];
ry(-3.0250149153379637) q[1];
cx q[0],q[1];
ry(-0.7734164490778932) q[0];
ry(2.773413720177725) q[1];
cx q[0],q[1];
ry(0.34985131022298066) q[2];
ry(0.3591163506589439) q[3];
cx q[2],q[3];
ry(1.0773222732276044) q[2];
ry(-2.7676566785832004) q[3];
cx q[2],q[3];
ry(-1.0694115785680673) q[4];
ry(2.3139732989638127) q[5];
cx q[4],q[5];
ry(2.844428743106055) q[4];
ry(1.8090773552714348) q[5];
cx q[4],q[5];
ry(0.13704792542689537) q[6];
ry(-2.0190376535396006) q[7];
cx q[6],q[7];
ry(1.5724682365474187) q[6];
ry(-0.7182629978253426) q[7];
cx q[6],q[7];
ry(-1.6255035359086207) q[8];
ry(0.7562087687777639) q[9];
cx q[8],q[9];
ry(0.06986835797124513) q[8];
ry(-1.3077122446009235) q[9];
cx q[8],q[9];
ry(1.4679418782784823) q[10];
ry(-2.818189484063654) q[11];
cx q[10],q[11];
ry(-1.6380380663321619) q[10];
ry(1.440836609359732) q[11];
cx q[10],q[11];
ry(2.406380559717497) q[12];
ry(1.4885433580963474) q[13];
cx q[12],q[13];
ry(-0.3972607231731389) q[12];
ry(-0.5972050919581183) q[13];
cx q[12],q[13];
ry(2.047751327713187) q[14];
ry(1.6064512414559875) q[15];
cx q[14],q[15];
ry(-2.0645919286575216) q[14];
ry(-1.7980858904355133) q[15];
cx q[14],q[15];
ry(3.056826490886066) q[16];
ry(0.813351524854415) q[17];
cx q[16],q[17];
ry(-2.377003909901742) q[16];
ry(-2.3650351611875813) q[17];
cx q[16],q[17];
ry(0.5039838371926608) q[18];
ry(1.1604465491992209) q[19];
cx q[18],q[19];
ry(-0.21542385814328066) q[18];
ry(-0.03134114191589238) q[19];
cx q[18],q[19];
ry(2.0612639250456377) q[0];
ry(-2.788501769360671) q[2];
cx q[0],q[2];
ry(-2.5692700453526824) q[0];
ry(2.8945818350544874) q[2];
cx q[0],q[2];
ry(-0.5337179004989396) q[2];
ry(1.516462652061998) q[4];
cx q[2],q[4];
ry(0.8869897076036022) q[2];
ry(-0.5649574735732941) q[4];
cx q[2],q[4];
ry(3.0946980987421107) q[4];
ry(-2.9387822177707976) q[6];
cx q[4],q[6];
ry(-0.09792274083532387) q[4];
ry(-3.1208017149863156) q[6];
cx q[4],q[6];
ry(1.0689398256030884) q[6];
ry(2.9804329367669222) q[8];
cx q[6],q[8];
ry(3.1373976839553297) q[6];
ry(3.1401948980805865) q[8];
cx q[6],q[8];
ry(-1.7966487036508654) q[8];
ry(1.4927326138701629) q[10];
cx q[8],q[10];
ry(-2.148903097637521) q[8];
ry(-1.470698287133235) q[10];
cx q[8],q[10];
ry(-1.6516200557566632) q[10];
ry(2.333619896155815) q[12];
cx q[10],q[12];
ry(-3.1404519010356378) q[10];
ry(3.139399676648765) q[12];
cx q[10],q[12];
ry(2.4570725604685064) q[12];
ry(-3.0050869024295115) q[14];
cx q[12],q[14];
ry(-2.8375785399507416) q[12];
ry(2.3099366925791287) q[14];
cx q[12],q[14];
ry(-0.2672805114400658) q[14];
ry(-1.0386418863486282) q[16];
cx q[14],q[16];
ry(0.005945131731704301) q[14];
ry(-0.008976045350557627) q[16];
cx q[14],q[16];
ry(-0.02080023436413591) q[16];
ry(0.8851885981459134) q[18];
cx q[16],q[18];
ry(0.6679219767799927) q[16];
ry(-0.019988762309503194) q[18];
cx q[16],q[18];
ry(-1.7761561306501437) q[1];
ry(-0.03944748329133218) q[3];
cx q[1],q[3];
ry(-0.32201376133439374) q[1];
ry(-0.41658326696189246) q[3];
cx q[1],q[3];
ry(3.0069154219397136) q[3];
ry(-1.4627220778230368) q[5];
cx q[3],q[5];
ry(0.3575455783035979) q[3];
ry(1.8005200180053054) q[5];
cx q[3],q[5];
ry(0.24223771631416616) q[5];
ry(2.463998653468122) q[7];
cx q[5],q[7];
ry(-3.1290332526660154) q[5];
ry(-3.09247143086922) q[7];
cx q[5],q[7];
ry(-2.3145637260837217) q[7];
ry(-0.8338963183559936) q[9];
cx q[7],q[9];
ry(-0.5543825098877491) q[7];
ry(-3.136458493874968) q[9];
cx q[7],q[9];
ry(3.1096892698795857) q[9];
ry(-0.1379454429497997) q[11];
cx q[9],q[11];
ry(1.594520252530642) q[9];
ry(-1.6063959329670006) q[11];
cx q[9],q[11];
ry(-2.6271913782099423) q[11];
ry(-1.216182683158142) q[13];
cx q[11],q[13];
ry(-2.470502678309867) q[11];
ry(-3.122475525734352) q[13];
cx q[11],q[13];
ry(-2.174923449658083) q[13];
ry(-0.02386581837433236) q[15];
cx q[13],q[15];
ry(-2.658618900746178) q[13];
ry(2.4139942189082397) q[15];
cx q[13],q[15];
ry(-1.062874167172148) q[15];
ry(1.469052897244607) q[17];
cx q[15],q[17];
ry(3.126066300318418) q[15];
ry(-3.136036792049196) q[17];
cx q[15],q[17];
ry(2.7447161717908757) q[17];
ry(1.049673158230757) q[19];
cx q[17],q[19];
ry(2.9336257533028367) q[17];
ry(-1.8501253266954507) q[19];
cx q[17],q[19];
ry(0.47180505008000573) q[0];
ry(0.26774142355669195) q[1];
cx q[0],q[1];
ry(-3.021944642486181) q[0];
ry(-2.624750769107464) q[1];
cx q[0],q[1];
ry(-1.734460333520019) q[2];
ry(1.7090160610468885) q[3];
cx q[2],q[3];
ry(-2.8865959977082443) q[2];
ry(-0.19641205091648647) q[3];
cx q[2],q[3];
ry(-1.904498975801284) q[4];
ry(-2.8306906042294875) q[5];
cx q[4],q[5];
ry(1.6202251094016715) q[4];
ry(-0.4435848319283633) q[5];
cx q[4],q[5];
ry(-0.8665726503407649) q[6];
ry(1.3561672999318768) q[7];
cx q[6],q[7];
ry(-0.0017362413380983535) q[6];
ry(-1.0838503630566336) q[7];
cx q[6],q[7];
ry(-0.39701617661759947) q[8];
ry(-2.8822333691375626) q[9];
cx q[8],q[9];
ry(2.7113021362551937) q[8];
ry(0.08458391373765856) q[9];
cx q[8],q[9];
ry(-1.6253821105693926) q[10];
ry(-1.466071726928056) q[11];
cx q[10],q[11];
ry(-1.5808454546540995) q[10];
ry(-1.5727344287938685) q[11];
cx q[10],q[11];
ry(-0.7442649200185371) q[12];
ry(2.6732424252157334) q[13];
cx q[12],q[13];
ry(1.3456623024610683) q[12];
ry(-1.6471500170123745) q[13];
cx q[12],q[13];
ry(-2.90165674938254) q[14];
ry(-0.3836524781420949) q[15];
cx q[14],q[15];
ry(-1.4179177603511244) q[14];
ry(2.739535924518947) q[15];
cx q[14],q[15];
ry(0.7348233003711632) q[16];
ry(-2.66690612727583) q[17];
cx q[16],q[17];
ry(-1.7387548319160215) q[16];
ry(0.28916856719090145) q[17];
cx q[16],q[17];
ry(-0.946389454580431) q[18];
ry(-1.3896124162311367) q[19];
cx q[18],q[19];
ry(-2.738955391880604) q[18];
ry(1.633208115781711) q[19];
cx q[18],q[19];
ry(1.4313259146099373) q[0];
ry(-2.374592655156375) q[2];
cx q[0],q[2];
ry(-1.2874404870499765) q[0];
ry(2.6897538637622986) q[2];
cx q[0],q[2];
ry(-1.5899237635739916) q[2];
ry(1.0015797318487896) q[4];
cx q[2],q[4];
ry(0.05094442347506156) q[2];
ry(0.30559624247480677) q[4];
cx q[2],q[4];
ry(2.5109212837507) q[4];
ry(-0.4630942331824235) q[6];
cx q[4],q[6];
ry(0.129395269972707) q[4];
ry(3.135972808281031) q[6];
cx q[4],q[6];
ry(-1.7553455642728393) q[6];
ry(-1.59637655325818) q[8];
cx q[6],q[8];
ry(3.1402036426911213) q[6];
ry(2.71328341177468) q[8];
cx q[6],q[8];
ry(-2.0530863040306064) q[8];
ry(-0.007497212052088642) q[10];
cx q[8],q[10];
ry(-1.5464276230912404) q[8];
ry(1.5660816742025556) q[10];
cx q[8],q[10];
ry(-1.482339731099339) q[10];
ry(2.689388648861374) q[12];
cx q[10],q[12];
ry(-0.0800869757006284) q[10];
ry(-0.03143233294981762) q[12];
cx q[10],q[12];
ry(-1.0791739340595734) q[12];
ry(-2.829203714473155) q[14];
cx q[12],q[14];
ry(-1.6949234984909516) q[12];
ry(-1.3865155837679566) q[14];
cx q[12],q[14];
ry(3.027119879312373) q[14];
ry(0.5444450978717859) q[16];
cx q[14],q[16];
ry(-3.1385641628232674) q[14];
ry(3.1316444688195575) q[16];
cx q[14],q[16];
ry(0.21462905883028327) q[16];
ry(-1.4544186791541565) q[18];
cx q[16],q[18];
ry(-1.0806053319254059) q[16];
ry(-2.7283667877910123) q[18];
cx q[16],q[18];
ry(1.315473234043374) q[1];
ry(1.633310844187314) q[3];
cx q[1],q[3];
ry(0.26663326925133324) q[1];
ry(-2.246523830856057) q[3];
cx q[1],q[3];
ry(-1.7437761165411212) q[3];
ry(1.2982727290708214) q[5];
cx q[3],q[5];
ry(2.4160816576449937) q[3];
ry(-0.11047305332770636) q[5];
cx q[3],q[5];
ry(0.06516116359202684) q[5];
ry(-1.4541234773639773) q[7];
cx q[5],q[7];
ry(0.005058293409554437) q[5];
ry(0.017540859808635376) q[7];
cx q[5],q[7];
ry(-1.7857477098178478) q[7];
ry(2.9792469902764482) q[9];
cx q[7],q[9];
ry(-0.0036845023604797746) q[7];
ry(0.006552586632471602) q[9];
cx q[7],q[9];
ry(-2.062312284487854) q[9];
ry(1.9714674979662918) q[11];
cx q[9],q[11];
ry(-0.048467572855698186) q[9];
ry(1.556572852558558) q[11];
cx q[9],q[11];
ry(-1.8782207887452467) q[11];
ry(2.2959602998826263) q[13];
cx q[11],q[13];
ry(0.0025686555650368623) q[11];
ry(3.13850956707511) q[13];
cx q[11],q[13];
ry(-2.750178674349327) q[13];
ry(-0.7520760135917235) q[15];
cx q[13],q[15];
ry(-2.821997898893101) q[13];
ry(-2.6663635706497937) q[15];
cx q[13],q[15];
ry(1.698484193674032) q[15];
ry(-1.4839245867185151) q[17];
cx q[15],q[17];
ry(0.005052941113822707) q[15];
ry(-3.1353961777139507) q[17];
cx q[15],q[17];
ry(-0.7475926812914739) q[17];
ry(1.2636639732534665) q[19];
cx q[17],q[19];
ry(-1.6560170458993677) q[17];
ry(-0.1478645003290868) q[19];
cx q[17],q[19];
ry(2.2198056972986007) q[0];
ry(1.4969246435281451) q[1];
cx q[0],q[1];
ry(1.6242058130669037) q[0];
ry(-0.36531209465106773) q[1];
cx q[0],q[1];
ry(1.2102282033593754) q[2];
ry(2.3663094834003795) q[3];
cx q[2],q[3];
ry(-1.8121387542621206) q[2];
ry(-2.2251338500713445) q[3];
cx q[2],q[3];
ry(-0.05641761098522746) q[4];
ry(-2.5904006140516302) q[5];
cx q[4],q[5];
ry(-0.40053997726940427) q[4];
ry(2.7000470071124765) q[5];
cx q[4],q[5];
ry(1.1683749531775647) q[6];
ry(-2.6239442691851025) q[7];
cx q[6],q[7];
ry(-3.1409657725527125) q[6];
ry(0.8005144536107531) q[7];
cx q[6],q[7];
ry(1.8688895529786116) q[8];
ry(-2.9141412899922385) q[9];
cx q[8],q[9];
ry(1.595391159068783) q[8];
ry(-0.04793017326452986) q[9];
cx q[8],q[9];
ry(2.3794353901765577) q[10];
ry(-2.8891625783235577) q[11];
cx q[10],q[11];
ry(1.5393819206137744) q[10];
ry(-3.12045183512554) q[11];
cx q[10],q[11];
ry(-0.24984615499404225) q[12];
ry(-0.3405437610526007) q[13];
cx q[12],q[13];
ry(-0.10477716665855602) q[12];
ry(-2.999237506996764) q[13];
cx q[12],q[13];
ry(1.409846194151152) q[14];
ry(-2.637222846725235) q[15];
cx q[14],q[15];
ry(1.5335182813664279) q[14];
ry(1.5812311425265346) q[15];
cx q[14],q[15];
ry(1.3298186782306738) q[16];
ry(-3.118158504092833) q[17];
cx q[16],q[17];
ry(-0.8426993846977497) q[16];
ry(2.2533969828706484) q[17];
cx q[16],q[17];
ry(0.3886875773707352) q[18];
ry(-1.3741378997312) q[19];
cx q[18],q[19];
ry(-2.9925063895372515) q[18];
ry(1.9769895763180534) q[19];
cx q[18],q[19];
ry(2.4807071434908052) q[0];
ry(2.2614755162123936) q[2];
cx q[0],q[2];
ry(-0.6042364611900486) q[0];
ry(1.124871331274135) q[2];
cx q[0],q[2];
ry(-1.8061519183219201) q[2];
ry(1.570992792627217) q[4];
cx q[2],q[4];
ry(-0.24826674649090563) q[2];
ry(0.016683990343183016) q[4];
cx q[2],q[4];
ry(-2.242436514748827) q[4];
ry(0.851048382053599) q[6];
cx q[4],q[6];
ry(3.088314406662032) q[4];
ry(-3.1290387613833333) q[6];
cx q[4],q[6];
ry(-2.548816422306576) q[6];
ry(1.5605929988508913) q[8];
cx q[6],q[8];
ry(0.012099810950762624) q[6];
ry(-3.1414004930935935) q[8];
cx q[6],q[8];
ry(0.7937214160191022) q[8];
ry(2.742675824288095) q[10];
cx q[8],q[10];
ry(0.007628067090642787) q[8];
ry(3.1378092958175934) q[10];
cx q[8],q[10];
ry(-3.127459595996731) q[10];
ry(-1.9399827488433097) q[12];
cx q[10],q[12];
ry(-2.8701181490028005) q[10];
ry(-0.003566124323389985) q[12];
cx q[10],q[12];
ry(1.4861769401838973) q[12];
ry(0.14417341920849877) q[14];
cx q[12],q[14];
ry(-2.0342976284565535) q[12];
ry(-1.682666268830947) q[14];
cx q[12],q[14];
ry(-0.07846272619889838) q[14];
ry(0.41411543916348226) q[16];
cx q[14],q[16];
ry(0.16135881616713382) q[14];
ry(2.50512053510505) q[16];
cx q[14],q[16];
ry(-1.571124907832993) q[16];
ry(-0.4634141312177738) q[18];
cx q[16],q[18];
ry(1.3614030861902506) q[16];
ry(0.5732370860628233) q[18];
cx q[16],q[18];
ry(-1.1282464584445737) q[1];
ry(-1.3560044174225425) q[3];
cx q[1],q[3];
ry(2.384899261246244) q[1];
ry(-1.2253403574623594) q[3];
cx q[1],q[3];
ry(1.7560273791906102) q[3];
ry(-0.9387929852731371) q[5];
cx q[3],q[5];
ry(2.903631719517539) q[3];
ry(-3.0794662720837476) q[5];
cx q[3],q[5];
ry(0.6562634555848826) q[5];
ry(2.128397436653434) q[7];
cx q[5],q[7];
ry(3.140222036023883) q[5];
ry(0.002833222126452206) q[7];
cx q[5],q[7];
ry(1.9489377959644445) q[7];
ry(1.211526109185657) q[9];
cx q[7],q[9];
ry(2.8272834090239067) q[7];
ry(-0.000912867403488348) q[9];
cx q[7],q[9];
ry(-1.9535550446478167) q[9];
ry(1.740388323440699) q[11];
cx q[9],q[11];
ry(3.005108583389667) q[9];
ry(-0.07294734458048602) q[11];
cx q[9],q[11];
ry(2.7136893436358114) q[11];
ry(-1.6748604370586975) q[13];
cx q[11],q[13];
ry(-0.019605406112874854) q[11];
ry(-0.0065494937663705954) q[13];
cx q[11],q[13];
ry(1.2515313094253262) q[13];
ry(0.6944806908144362) q[15];
cx q[13],q[15];
ry(3.070058711655542) q[13];
ry(3.0794270567500015) q[15];
cx q[13],q[15];
ry(2.5840548246101323) q[15];
ry(-1.3064448960748702) q[17];
cx q[15],q[17];
ry(0.7468560808593964) q[15];
ry(-0.48554088712229787) q[17];
cx q[15],q[17];
ry(3.1363390446499593) q[17];
ry(1.2571872718315555) q[19];
cx q[17],q[19];
ry(-2.4074732276748994) q[17];
ry(-1.5821379530313484) q[19];
cx q[17],q[19];
ry(0.8731473337355997) q[0];
ry(-3.0620494143137504) q[1];
cx q[0],q[1];
ry(-1.167005085648949) q[0];
ry(2.6034284467622033) q[1];
cx q[0],q[1];
ry(2.1931998722875) q[2];
ry(-1.384662336235372) q[3];
cx q[2],q[3];
ry(0.38675719916113244) q[2];
ry(-0.9110440682366701) q[3];
cx q[2],q[3];
ry(0.5556997362946268) q[4];
ry(2.561896194550544) q[5];
cx q[4],q[5];
ry(-1.9963055648147023) q[4];
ry(1.0415261096289332) q[5];
cx q[4],q[5];
ry(2.1692285522841956) q[6];
ry(-0.3123791626215749) q[7];
cx q[6],q[7];
ry(0.0005752833580716654) q[6];
ry(1.2170819731046194) q[7];
cx q[6],q[7];
ry(-2.333036898074965) q[8];
ry(-1.3910059620867798) q[9];
cx q[8],q[9];
ry(0.22484415779680386) q[8];
ry(-0.02290793839867287) q[9];
cx q[8],q[9];
ry(1.3194311106976102) q[10];
ry(0.5813264486067515) q[11];
cx q[10],q[11];
ry(1.7297338500455082) q[10];
ry(1.553182192468826) q[11];
cx q[10],q[11];
ry(2.344584527597685) q[12];
ry(-2.338564908027369) q[13];
cx q[12],q[13];
ry(0.48674998280100745) q[12];
ry(-0.5803231209281723) q[13];
cx q[12],q[13];
ry(-1.5009554580106528) q[14];
ry(-1.3395894899515115) q[15];
cx q[14],q[15];
ry(2.962103014543683) q[14];
ry(2.2692822474979404) q[15];
cx q[14],q[15];
ry(-0.004300494706458658) q[16];
ry(-1.1782834086648872) q[17];
cx q[16],q[17];
ry(-3.092620640249876) q[16];
ry(1.1157047494207075) q[17];
cx q[16],q[17];
ry(-2.7376736975282796) q[18];
ry(1.4364990745485027) q[19];
cx q[18],q[19];
ry(-3.1279028363068364) q[18];
ry(3.1155057699323985) q[19];
cx q[18],q[19];
ry(0.3678391328356995) q[0];
ry(-0.8038578653524349) q[2];
cx q[0],q[2];
ry(-1.8565917823274862) q[0];
ry(-1.4683657414675597) q[2];
cx q[0],q[2];
ry(-1.588523757069592) q[2];
ry(-2.4795914890813453) q[4];
cx q[2],q[4];
ry(-0.06770885564393228) q[2];
ry(-2.9376244764534465) q[4];
cx q[2],q[4];
ry(-0.3322040334308308) q[4];
ry(-2.4157076141647407) q[6];
cx q[4],q[6];
ry(-2.994482445862024) q[4];
ry(-0.1456071067712048) q[6];
cx q[4],q[6];
ry(1.7283332647336425) q[6];
ry(1.3198373027658041) q[8];
cx q[6],q[8];
ry(-0.0015896674413495818) q[6];
ry(3.138914733840072) q[8];
cx q[6],q[8];
ry(-1.5253099282435043) q[8];
ry(-2.799310449284866) q[10];
cx q[8],q[10];
ry(0.003798912091004696) q[8];
ry(-3.110751265326569) q[10];
cx q[8],q[10];
ry(1.2976717960111759) q[10];
ry(1.4925797661949356) q[12];
cx q[10],q[12];
ry(0.0053084551823890215) q[10];
ry(-0.007079261282342323) q[12];
cx q[10],q[12];
ry(2.520408757296905) q[12];
ry(1.5386661832243644) q[14];
cx q[12],q[14];
ry(-0.0018476704750603069) q[12];
ry(-3.128258813866135) q[14];
cx q[12],q[14];
ry(-3.0862532475232074) q[14];
ry(1.6040455701290692) q[16];
cx q[14],q[16];
ry(-0.015334106852055386) q[14];
ry(0.00564364776490045) q[16];
cx q[14],q[16];
ry(3.099392794324736) q[16];
ry(-1.1734156256265682) q[18];
cx q[16],q[18];
ry(-1.5522416650446145) q[16];
ry(1.5861430270234143) q[18];
cx q[16],q[18];
ry(-1.5165937243406153) q[1];
ry(1.8928543550114374) q[3];
cx q[1],q[3];
ry(-2.8203969576630064) q[1];
ry(2.0166251987331223) q[3];
cx q[1],q[3];
ry(-1.899010985377818) q[3];
ry(0.14845971473536768) q[5];
cx q[3],q[5];
ry(3.0008678984581434) q[3];
ry(-0.09334683498338855) q[5];
cx q[3],q[5];
ry(-1.7805195704130554) q[5];
ry(-1.6577385085616054) q[7];
cx q[5],q[7];
ry(-0.00441027774469741) q[5];
ry(0.0199480577677947) q[7];
cx q[5],q[7];
ry(-2.845245773139695) q[7];
ry(-2.9108513234062077) q[9];
cx q[7],q[9];
ry(2.8249900988832732) q[7];
ry(-2.722787684772086) q[9];
cx q[7],q[9];
ry(-1.1547755016087304) q[9];
ry(3.127583679401561) q[11];
cx q[9],q[11];
ry(1.758215188412497) q[9];
ry(1.574287101714924) q[11];
cx q[9],q[11];
ry(2.477425341946313) q[11];
ry(0.9515949420371346) q[13];
cx q[11],q[13];
ry(-3.1335247097140853) q[11];
ry(-0.0029495485375995592) q[13];
cx q[11],q[13];
ry(-2.5295460514277717) q[13];
ry(-0.576349367261444) q[15];
cx q[13],q[15];
ry(0.011291855087473657) q[13];
ry(3.137137842121106) q[15];
cx q[13],q[15];
ry(-2.1211482940796715) q[15];
ry(0.6466775032465295) q[17];
cx q[15],q[17];
ry(3.1410196331269873) q[15];
ry(-3.1356037509231376) q[17];
cx q[15],q[17];
ry(-0.9627118482490458) q[17];
ry(1.5830177110028245) q[19];
cx q[17],q[19];
ry(-0.8364023362537889) q[17];
ry(3.1292949845352527) q[19];
cx q[17],q[19];
ry(0.14413489206258223) q[0];
ry(-3.114849342867876) q[1];
cx q[0],q[1];
ry(1.4585399643950465) q[0];
ry(2.252099367962557) q[1];
cx q[0],q[1];
ry(-2.9113717795315464) q[2];
ry(-2.7603681109981637) q[3];
cx q[2],q[3];
ry(-1.5546255768871764) q[2];
ry(1.6652972628572797) q[3];
cx q[2],q[3];
ry(0.20488763015031514) q[4];
ry(0.5299256814736222) q[5];
cx q[4],q[5];
ry(1.651000204811643) q[4];
ry(-2.681710369700691) q[5];
cx q[4],q[5];
ry(1.8037018603938506) q[6];
ry(-1.5757327484734285) q[7];
cx q[6],q[7];
ry(1.5665407124276767) q[6];
ry(1.5718608607318743) q[7];
cx q[6],q[7];
ry(2.6885334977845132) q[8];
ry(0.02849418814054871) q[9];
cx q[8],q[9];
ry(-1.6004953226271734) q[8];
ry(1.1242691194317471) q[9];
cx q[8],q[9];
ry(1.5281147645370199) q[10];
ry(-1.004335977129582) q[11];
cx q[10],q[11];
ry(-1.5844101381276776) q[10];
ry(-1.43894623352107) q[11];
cx q[10],q[11];
ry(-2.6633978795987856) q[12];
ry(-1.5016141479244776) q[13];
cx q[12],q[13];
ry(-2.9256118225243766) q[12];
ry(0.85333745813079) q[13];
cx q[12],q[13];
ry(1.6457983568707046) q[14];
ry(-2.926703717893704) q[15];
cx q[14],q[15];
ry(-1.7413400725682409) q[14];
ry(-0.6977977037547479) q[15];
cx q[14],q[15];
ry(0.8894641715874733) q[16];
ry(1.9262007283784657) q[17];
cx q[16],q[17];
ry(-0.08830905633877961) q[16];
ry(0.11728144872857682) q[17];
cx q[16],q[17];
ry(-0.20192900618514376) q[18];
ry(2.2847552442627035) q[19];
cx q[18],q[19];
ry(1.5731081586886635) q[18];
ry(-1.5701085810819837) q[19];
cx q[18],q[19];
ry(2.8638570666418675) q[0];
ry(-1.3736388735089844) q[2];
cx q[0],q[2];
ry(1.4686263764794603) q[0];
ry(1.5814883763738379) q[2];
cx q[0],q[2];
ry(-2.2406472724884) q[2];
ry(-2.7264331985366455) q[4];
cx q[2],q[4];
ry(-0.1890937471309773) q[2];
ry(-0.20182668632694573) q[4];
cx q[2],q[4];
ry(0.910343877016096) q[4];
ry(-1.640455969208894) q[6];
cx q[4],q[6];
ry(3.090243315183265) q[4];
ry(0.021030932364006997) q[6];
cx q[4],q[6];
ry(1.51568848024219) q[6];
ry(1.9374088624995631) q[8];
cx q[6],q[8];
ry(0.0025161155851014527) q[6];
ry(-0.043371287550787396) q[8];
cx q[6],q[8];
ry(2.7305577035359248) q[8];
ry(0.11782252023634587) q[10];
cx q[8],q[10];
ry(3.117190196388566) q[8];
ry(-3.0483326219562024) q[10];
cx q[8],q[10];
ry(0.41625017029086403) q[10];
ry(-0.7697130656211684) q[12];
cx q[10],q[12];
ry(-0.02930960007497521) q[10];
ry(0.014502890637153598) q[12];
cx q[10],q[12];
ry(1.3102662298697583) q[12];
ry(2.5990377769464197) q[14];
cx q[12],q[14];
ry(0.008476915813224137) q[12];
ry(-0.04822207190746985) q[14];
cx q[12],q[14];
ry(0.9672225487544873) q[14];
ry(2.9642937144544423) q[16];
cx q[14],q[16];
ry(3.1315252992385343) q[14];
ry(0.0022903509563634737) q[16];
cx q[14],q[16];
ry(1.6295482638911218) q[16];
ry(0.011230710090239349) q[18];
cx q[16],q[18];
ry(3.0340943366061173) q[16];
ry(1.582245467122294) q[18];
cx q[16],q[18];
ry(-0.26635430413299077) q[1];
ry(-1.5952926750465144) q[3];
cx q[1],q[3];
ry(-1.3418292555723488) q[1];
ry(-1.072543735164045) q[3];
cx q[1],q[3];
ry(-1.4647440955947957) q[3];
ry(0.04038394982545927) q[5];
cx q[3],q[5];
ry(-0.16555814931720916) q[3];
ry(2.1268014850814776) q[5];
cx q[3],q[5];
ry(-1.7129852911602692) q[5];
ry(1.5657198958487442) q[7];
cx q[5],q[7];
ry(2.7939008342911076) q[5];
ry(0.04417087787315399) q[7];
cx q[5],q[7];
ry(1.721872051727907) q[7];
ry(-1.5662351123428113) q[9];
cx q[7],q[9];
ry(1.5640215725178805) q[7];
ry(-3.1411340461396007) q[9];
cx q[7],q[9];
ry(-0.3801362276854985) q[9];
ry(-1.498837875674311) q[11];
cx q[9],q[11];
ry(1.5717270225530477) q[9];
ry(3.1398983172143877) q[11];
cx q[9],q[11];
ry(-0.9082155015175086) q[11];
ry(1.796716557307227) q[13];
cx q[11],q[13];
ry(-1.5678594838821682) q[11];
ry(-3.1398819204155513) q[13];
cx q[11],q[13];
ry(1.5690034858508637) q[13];
ry(-1.630417144258381) q[15];
cx q[13],q[15];
ry(1.5711501822056373) q[13];
ry(-0.04050819594838728) q[15];
cx q[13],q[15];
ry(0.34869988822481) q[15];
ry(-0.009599674354731746) q[17];
cx q[15],q[17];
ry(-1.5709013025306682) q[15];
ry(-0.0013248486367709005) q[17];
cx q[15],q[17];
ry(1.5690261123998743) q[17];
ry(1.5018630666121329) q[19];
cx q[17],q[19];
ry(1.570708408230768) q[17];
ry(-0.2064092676159106) q[19];
cx q[17],q[19];
ry(-2.950687168581171) q[0];
ry(2.1991388201620206) q[1];
ry(-1.4915429678660552) q[2];
ry(1.5826384424600626) q[3];
ry(0.46402877476662135) q[4];
ry(-1.544093656720749) q[5];
ry(-1.5894275144271066) q[6];
ry(-1.4233073284665314) q[7];
ry(-1.5882410653745174) q[8];
ry(2.761970077731528) q[9];
ry(-0.289373176375606) q[10];
ry(2.2336158261740486) q[11];
ry(-1.184452236704554) q[12];
ry(-1.5689692468291474) q[13];
ry(-2.0574080880648236) q[14];
ry(-0.34821242685938775) q[15];
ry(0.002184013649248944) q[16];
ry(-1.5687220828888013) q[17];
ry(0.02363603637499967) q[18];
ry(1.5704721025028627) q[19];