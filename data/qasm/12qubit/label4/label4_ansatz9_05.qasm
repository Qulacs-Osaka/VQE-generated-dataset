OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.1016583439048304) q[0];
ry(-2.9782835400150307) q[1];
cx q[0],q[1];
ry(-2.5848652884364482) q[0];
ry(1.7847024759592713) q[1];
cx q[0],q[1];
ry(-1.7912749512043256) q[2];
ry(-1.7917740099915582) q[3];
cx q[2],q[3];
ry(1.2942211623291682) q[2];
ry(2.783687229006461) q[3];
cx q[2],q[3];
ry(-1.9456394477651262) q[4];
ry(2.4816763497966194) q[5];
cx q[4],q[5];
ry(-1.836101504617267) q[4];
ry(2.6114032877239097) q[5];
cx q[4],q[5];
ry(2.0637897713947573) q[6];
ry(-2.764169922476047) q[7];
cx q[6],q[7];
ry(1.206602733422982) q[6];
ry(-3.095165709623529) q[7];
cx q[6],q[7];
ry(-1.3689424431523112) q[8];
ry(2.3284981347011153) q[9];
cx q[8],q[9];
ry(0.4724132111163799) q[8];
ry(-0.6078902377276939) q[9];
cx q[8],q[9];
ry(-1.0437664628388565) q[10];
ry(-0.8826632433010033) q[11];
cx q[10],q[11];
ry(-1.5525255498099646) q[10];
ry(2.451514448238566) q[11];
cx q[10],q[11];
ry(-1.9692554186252211) q[0];
ry(-1.2328708548961749) q[2];
cx q[0],q[2];
ry(-0.47396706050372295) q[0];
ry(1.604815459583373) q[2];
cx q[0],q[2];
ry(-1.9161174505552347) q[2];
ry(-0.9172552560346734) q[4];
cx q[2],q[4];
ry(-2.2709769923356307) q[2];
ry(1.8465204443910979) q[4];
cx q[2],q[4];
ry(-1.5024895027657246) q[4];
ry(2.249770755072624) q[6];
cx q[4],q[6];
ry(1.976442606719679) q[4];
ry(-1.5019779573730512) q[6];
cx q[4],q[6];
ry(-1.7758568609867191) q[6];
ry(2.3942364437525145) q[8];
cx q[6],q[8];
ry(-2.116605070965506) q[6];
ry(0.4463100478153867) q[8];
cx q[6],q[8];
ry(0.6190365264848817) q[8];
ry(-0.06887667872168102) q[10];
cx q[8],q[10];
ry(1.3397401970666898) q[8];
ry(-0.3013242440016768) q[10];
cx q[8],q[10];
ry(-0.12284734696354427) q[1];
ry(0.7702479060982483) q[3];
cx q[1],q[3];
ry(-2.017487725870832) q[1];
ry(-2.004771976879483) q[3];
cx q[1],q[3];
ry(0.512380049302936) q[3];
ry(-1.467603424706308) q[5];
cx q[3],q[5];
ry(-2.4977651568384154) q[3];
ry(1.229885213818859) q[5];
cx q[3],q[5];
ry(2.881171293412594) q[5];
ry(2.9339930314072724) q[7];
cx q[5],q[7];
ry(-2.6943021520075194) q[5];
ry(-1.6405440694188809) q[7];
cx q[5],q[7];
ry(2.3646852615414633) q[7];
ry(0.7624520738099536) q[9];
cx q[7],q[9];
ry(0.7523211880264391) q[7];
ry(2.5191590415883582) q[9];
cx q[7],q[9];
ry(-2.090668533507441) q[9];
ry(-0.9383719858164784) q[11];
cx q[9],q[11];
ry(-1.9564421596273924) q[9];
ry(0.8030183705219399) q[11];
cx q[9],q[11];
ry(0.22808823334169018) q[0];
ry(-0.3749822739573269) q[3];
cx q[0],q[3];
ry(-1.808180587205757) q[0];
ry(2.1739367884592955) q[3];
cx q[0],q[3];
ry(0.6527591543211319) q[1];
ry(2.1022753904980522) q[2];
cx q[1],q[2];
ry(1.014882135579354) q[1];
ry(2.6117456991102066) q[2];
cx q[1],q[2];
ry(-1.5028142831381341) q[2];
ry(0.7890571194505942) q[5];
cx q[2],q[5];
ry(-1.2668651537447724) q[2];
ry(-2.424769512514443) q[5];
cx q[2],q[5];
ry(1.4337775074454573) q[3];
ry(-0.9651020314014461) q[4];
cx q[3],q[4];
ry(-2.341559523206431) q[3];
ry(-0.6596458565186314) q[4];
cx q[3],q[4];
ry(2.1675095494364673) q[4];
ry(0.2763452649435285) q[7];
cx q[4],q[7];
ry(-2.322671878158375) q[4];
ry(-0.7710496433567895) q[7];
cx q[4],q[7];
ry(-0.5133874753338147) q[5];
ry(0.8810297971713472) q[6];
cx q[5],q[6];
ry(-2.0973761158621205) q[5];
ry(1.7414193461407028) q[6];
cx q[5],q[6];
ry(1.4767559475829337) q[6];
ry(0.1504648471307019) q[9];
cx q[6],q[9];
ry(1.4380748552066507) q[6];
ry(0.9669645708541134) q[9];
cx q[6],q[9];
ry(-2.110639254617812) q[7];
ry(-1.172336605516336) q[8];
cx q[7],q[8];
ry(1.8105394878130294) q[7];
ry(0.4549272921112806) q[8];
cx q[7],q[8];
ry(-0.976106278480286) q[8];
ry(-0.5768313130548597) q[11];
cx q[8],q[11];
ry(-2.4378575158441143) q[8];
ry(1.9743187008737122) q[11];
cx q[8],q[11];
ry(1.2610889168657966) q[9];
ry(1.8089747577142958) q[10];
cx q[9],q[10];
ry(-0.4924462147706833) q[9];
ry(2.571759855893213) q[10];
cx q[9],q[10];
ry(-1.937371206647893) q[0];
ry(2.5486216057194553) q[1];
cx q[0],q[1];
ry(-0.31469383410549234) q[0];
ry(-1.2606240798370587) q[1];
cx q[0],q[1];
ry(-1.0938853135765692) q[2];
ry(2.300313609277979) q[3];
cx q[2],q[3];
ry(-0.7679510187956975) q[2];
ry(-2.5274991344950726) q[3];
cx q[2],q[3];
ry(0.12377531240745253) q[4];
ry(-2.096172265033053) q[5];
cx q[4],q[5];
ry(0.2815638511561566) q[4];
ry(0.6913869554078671) q[5];
cx q[4],q[5];
ry(2.921790798161696) q[6];
ry(3.1409339179952807) q[7];
cx q[6],q[7];
ry(2.845773480327905) q[6];
ry(-0.5833501791471551) q[7];
cx q[6],q[7];
ry(-1.6720921375429023) q[8];
ry(-0.7362513201809895) q[9];
cx q[8],q[9];
ry(2.4061482136819023) q[8];
ry(0.18425230278316995) q[9];
cx q[8],q[9];
ry(0.4935621700861304) q[10];
ry(2.7046814716316416) q[11];
cx q[10],q[11];
ry(1.1971908148394748) q[10];
ry(0.6990407031378819) q[11];
cx q[10],q[11];
ry(-1.020691181901868) q[0];
ry(-1.2848980367630907) q[2];
cx q[0],q[2];
ry(1.6528073699347399) q[0];
ry(-1.0239762679896085) q[2];
cx q[0],q[2];
ry(-0.6037844426064176) q[2];
ry(2.3282524818225703) q[4];
cx q[2],q[4];
ry(-1.9377253374107513) q[2];
ry(-1.687198741833237) q[4];
cx q[2],q[4];
ry(0.8997806791217754) q[4];
ry(-1.8530387595742928) q[6];
cx q[4],q[6];
ry(1.5633033665382232) q[4];
ry(-0.30876079213922225) q[6];
cx q[4],q[6];
ry(-3.017428284004267) q[6];
ry(-0.19891785071856827) q[8];
cx q[6],q[8];
ry(0.04384465522788538) q[6];
ry(-0.7856873793163892) q[8];
cx q[6],q[8];
ry(-0.724178080870896) q[8];
ry(2.7788244461628047) q[10];
cx q[8],q[10];
ry(0.8202316554721918) q[8];
ry(1.8158758475006094) q[10];
cx q[8],q[10];
ry(-1.277020098986926) q[1];
ry(-2.9640869544082578) q[3];
cx q[1],q[3];
ry(-2.8365260591950316) q[1];
ry(-0.975010693181835) q[3];
cx q[1],q[3];
ry(0.5105819829298595) q[3];
ry(-2.066348341094378) q[5];
cx q[3],q[5];
ry(-1.7763414028310933) q[3];
ry(0.6020501670828544) q[5];
cx q[3],q[5];
ry(1.58359995889984) q[5];
ry(-1.316176536810599) q[7];
cx q[5],q[7];
ry(0.5327458185578822) q[5];
ry(-1.1366977957995772) q[7];
cx q[5],q[7];
ry(-2.4284962340125067) q[7];
ry(-0.3726469457436225) q[9];
cx q[7],q[9];
ry(-0.14172922441708205) q[7];
ry(0.2453037411194714) q[9];
cx q[7],q[9];
ry(-1.8631207040027926) q[9];
ry(1.6255435193106935) q[11];
cx q[9],q[11];
ry(-0.2527900109739294) q[9];
ry(-2.1409361718364446) q[11];
cx q[9],q[11];
ry(1.5144093488141204) q[0];
ry(2.4949089778282802) q[3];
cx q[0],q[3];
ry(-0.8013147220966506) q[0];
ry(2.419932151278504) q[3];
cx q[0],q[3];
ry(0.02281762986518121) q[1];
ry(-0.9661974940069208) q[2];
cx q[1],q[2];
ry(0.6889995774124705) q[1];
ry(-1.9842546796817375) q[2];
cx q[1],q[2];
ry(2.8738822183652037) q[2];
ry(0.632179926190543) q[5];
cx q[2],q[5];
ry(0.35991714337081326) q[2];
ry(1.0593322754963035) q[5];
cx q[2],q[5];
ry(-0.4920364132172992) q[3];
ry(-2.652942182622455) q[4];
cx q[3],q[4];
ry(2.140527528148194) q[3];
ry(-2.6948970486651773) q[4];
cx q[3],q[4];
ry(-0.45334415105646514) q[4];
ry(2.836269714509658) q[7];
cx q[4],q[7];
ry(-1.5870902344021616) q[4];
ry(2.584247801593761) q[7];
cx q[4],q[7];
ry(-2.7034131877659164) q[5];
ry(1.1882849627181677) q[6];
cx q[5],q[6];
ry(-1.967949926832838) q[5];
ry(0.35158619330094437) q[6];
cx q[5],q[6];
ry(-2.0074463882573976) q[6];
ry(-1.1428506768173978) q[9];
cx q[6],q[9];
ry(-2.3507946446323227) q[6];
ry(0.888065278658333) q[9];
cx q[6],q[9];
ry(-0.46264030063053807) q[7];
ry(2.6351205536783153) q[8];
cx q[7],q[8];
ry(-0.678672321989378) q[7];
ry(-1.3840302330950744) q[8];
cx q[7],q[8];
ry(-0.4285834720636259) q[8];
ry(-2.5433087630222015) q[11];
cx q[8],q[11];
ry(-1.7104911210642673) q[8];
ry(2.081761290884308) q[11];
cx q[8],q[11];
ry(-2.8840638274713175) q[9];
ry(-2.3286079662378816) q[10];
cx q[9],q[10];
ry(-1.838973572863667) q[9];
ry(0.5594788582245687) q[10];
cx q[9],q[10];
ry(-1.118185215167835) q[0];
ry(-2.2701816203399847) q[1];
cx q[0],q[1];
ry(1.1439843646357635) q[0];
ry(1.6250255375945588) q[1];
cx q[0],q[1];
ry(2.724290634670137) q[2];
ry(-2.181031592937206) q[3];
cx q[2],q[3];
ry(2.771940084687117) q[2];
ry(2.8515353018134273) q[3];
cx q[2],q[3];
ry(0.41636303300349553) q[4];
ry(-2.6071898382251177) q[5];
cx q[4],q[5];
ry(-2.8600790678975057) q[4];
ry(2.4063292002829924) q[5];
cx q[4],q[5];
ry(0.40677635976777804) q[6];
ry(-2.8379060781111938) q[7];
cx q[6],q[7];
ry(-1.1911476159666412) q[6];
ry(-2.953503985463704) q[7];
cx q[6],q[7];
ry(-2.2827362014836936) q[8];
ry(-2.2697002545372316) q[9];
cx q[8],q[9];
ry(1.4454505063248086) q[8];
ry(2.1146248859163834) q[9];
cx q[8],q[9];
ry(1.2751225980674477) q[10];
ry(2.602028438088929) q[11];
cx q[10],q[11];
ry(1.7858338669876348) q[10];
ry(-2.592961898966405) q[11];
cx q[10],q[11];
ry(0.032486177216745564) q[0];
ry(-3.073265333243918) q[2];
cx q[0],q[2];
ry(-2.037839005746567) q[0];
ry(-2.1410292596601597) q[2];
cx q[0],q[2];
ry(2.045670275442062) q[2];
ry(-1.1119433052459131) q[4];
cx q[2],q[4];
ry(-1.8215356639121651) q[2];
ry(1.971010222241623) q[4];
cx q[2],q[4];
ry(1.8317345352790781) q[4];
ry(0.4540687788519494) q[6];
cx q[4],q[6];
ry(-1.0907201022519057) q[4];
ry(1.5688267284871005) q[6];
cx q[4],q[6];
ry(-1.111078226590073) q[6];
ry(-1.1370681771502724) q[8];
cx q[6],q[8];
ry(1.8963020115925189) q[6];
ry(1.9372040269044923) q[8];
cx q[6],q[8];
ry(-1.0665609455194245) q[8];
ry(-2.5785887709696045) q[10];
cx q[8],q[10];
ry(-0.4902318836161106) q[8];
ry(-0.9083162816825995) q[10];
cx q[8],q[10];
ry(-1.240920195963949) q[1];
ry(-2.4465096027025215) q[3];
cx q[1],q[3];
ry(2.5819635347266217) q[1];
ry(-1.2151331258774956) q[3];
cx q[1],q[3];
ry(-2.634723286263957) q[3];
ry(-1.5404732783036534) q[5];
cx q[3],q[5];
ry(1.528743231006014) q[3];
ry(-1.3023450071790603) q[5];
cx q[3],q[5];
ry(-2.1821120673567593) q[5];
ry(-2.0946326583723716) q[7];
cx q[5],q[7];
ry(-1.5804237348951746) q[5];
ry(2.8879443259790216) q[7];
cx q[5],q[7];
ry(2.5443060182897645) q[7];
ry(2.709146738705226) q[9];
cx q[7],q[9];
ry(-2.524825589646706) q[7];
ry(-0.15554735320960814) q[9];
cx q[7],q[9];
ry(1.7716761544447117) q[9];
ry(2.5358862199568835) q[11];
cx q[9],q[11];
ry(1.2787360539084576) q[9];
ry(1.045917936434729) q[11];
cx q[9],q[11];
ry(1.2669322481422647) q[0];
ry(-0.2187254327815671) q[3];
cx q[0],q[3];
ry(1.4027803227253726) q[0];
ry(2.3926121454794376) q[3];
cx q[0],q[3];
ry(-2.876146018567179) q[1];
ry(-2.581266131071217) q[2];
cx q[1],q[2];
ry(-0.23274664330775302) q[1];
ry(0.44950970887514163) q[2];
cx q[1],q[2];
ry(2.6788263466367077) q[2];
ry(-1.958911927765563) q[5];
cx q[2],q[5];
ry(1.8097391133454632) q[2];
ry(1.7584950596222262) q[5];
cx q[2],q[5];
ry(1.0886208922895477) q[3];
ry(1.2381991775536987) q[4];
cx q[3],q[4];
ry(-0.5879117898319196) q[3];
ry(2.4343362902974723) q[4];
cx q[3],q[4];
ry(2.9785535059097574) q[4];
ry(-3.0793428044287454) q[7];
cx q[4],q[7];
ry(1.0504975220884072) q[4];
ry(2.183996482580718) q[7];
cx q[4],q[7];
ry(3.133340241501191) q[5];
ry(-1.9518221095030992) q[6];
cx q[5],q[6];
ry(-2.001264017789781) q[5];
ry(2.183399977981243) q[6];
cx q[5],q[6];
ry(-1.774710212380842) q[6];
ry(2.516873287213021) q[9];
cx q[6],q[9];
ry(0.35727056278583724) q[6];
ry(-0.8557435533005612) q[9];
cx q[6],q[9];
ry(0.9286102868868934) q[7];
ry(-2.6932024301750093) q[8];
cx q[7],q[8];
ry(0.5415343399673604) q[7];
ry(2.1151637280487163) q[8];
cx q[7],q[8];
ry(2.3413674781392637) q[8];
ry(2.853045438490415) q[11];
cx q[8],q[11];
ry(1.6914193980217753) q[8];
ry(0.7219682427042862) q[11];
cx q[8],q[11];
ry(-1.898908863108625) q[9];
ry(-3.0323839012206606) q[10];
cx q[9],q[10];
ry(-1.0751134585337816) q[9];
ry(0.8243615194483898) q[10];
cx q[9],q[10];
ry(-0.6203161907700094) q[0];
ry(0.9302896192110159) q[1];
cx q[0],q[1];
ry(0.9852118820245357) q[0];
ry(-2.524072635039126) q[1];
cx q[0],q[1];
ry(-2.2552128890152954) q[2];
ry(-2.7081743100852798) q[3];
cx q[2],q[3];
ry(2.773956673877899) q[2];
ry(-0.5053334133697618) q[3];
cx q[2],q[3];
ry(2.102598355367765) q[4];
ry(-1.1691169645413462) q[5];
cx q[4],q[5];
ry(2.7525277161317865) q[4];
ry(2.123471971180747) q[5];
cx q[4],q[5];
ry(-1.1429352416069962) q[6];
ry(-2.476540427499027) q[7];
cx q[6],q[7];
ry(2.1080495589211585) q[6];
ry(1.960455632988203) q[7];
cx q[6],q[7];
ry(-0.9603882281278793) q[8];
ry(-2.548126237140234) q[9];
cx q[8],q[9];
ry(-0.19777412279492437) q[8];
ry(-0.9608153024903282) q[9];
cx q[8],q[9];
ry(1.5549914330696) q[10];
ry(-2.8193820484373724) q[11];
cx q[10],q[11];
ry(2.2147044393170385) q[10];
ry(-0.4310824015789993) q[11];
cx q[10],q[11];
ry(-1.2465639793532048) q[0];
ry(0.966673409373052) q[2];
cx q[0],q[2];
ry(0.28717267533718915) q[0];
ry(0.08351046042318444) q[2];
cx q[0],q[2];
ry(-2.9695408637299137) q[2];
ry(1.7714397898000511) q[4];
cx q[2],q[4];
ry(0.8814830563111063) q[2];
ry(1.4443906715402628) q[4];
cx q[2],q[4];
ry(-3.031260322115987) q[4];
ry(-1.453433178762725) q[6];
cx q[4],q[6];
ry(1.1145349228606198) q[4];
ry(0.598873814986786) q[6];
cx q[4],q[6];
ry(0.4228954635702599) q[6];
ry(-1.6924263388039642) q[8];
cx q[6],q[8];
ry(2.505179233963704) q[6];
ry(2.4454464907498665) q[8];
cx q[6],q[8];
ry(-1.8255330155948415) q[8];
ry(2.46961454188042) q[10];
cx q[8],q[10];
ry(0.9637833778662905) q[8];
ry(-1.1693871368792195) q[10];
cx q[8],q[10];
ry(1.2963436983914036) q[1];
ry(0.9768177131826872) q[3];
cx q[1],q[3];
ry(-1.841470053213232) q[1];
ry(-2.0823214914355557) q[3];
cx q[1],q[3];
ry(2.157638629260016) q[3];
ry(2.0308453989320947) q[5];
cx q[3],q[5];
ry(-2.3047946082534607) q[3];
ry(-1.4002526587359538) q[5];
cx q[3],q[5];
ry(-0.5392137048247707) q[5];
ry(1.3638789598975318) q[7];
cx q[5],q[7];
ry(-0.017352059557292052) q[5];
ry(1.1046440109151767) q[7];
cx q[5],q[7];
ry(2.465119879409712) q[7];
ry(-1.4455343968507774) q[9];
cx q[7],q[9];
ry(0.42309956990729214) q[7];
ry(-1.232308028226142) q[9];
cx q[7],q[9];
ry(-0.21778064630229643) q[9];
ry(0.26809321002626607) q[11];
cx q[9],q[11];
ry(-0.8998059966565073) q[9];
ry(-1.5386943899200787) q[11];
cx q[9],q[11];
ry(2.3236131411679155) q[0];
ry(-0.26401744237620595) q[3];
cx q[0],q[3];
ry(-0.12889869279199306) q[0];
ry(-1.9807411452727486) q[3];
cx q[0],q[3];
ry(2.2297028140139012) q[1];
ry(0.8313225872606764) q[2];
cx q[1],q[2];
ry(-0.7756357871770944) q[1];
ry(-2.3141597560021823) q[2];
cx q[1],q[2];
ry(2.958531288399589) q[2];
ry(-1.0725794806657531) q[5];
cx q[2],q[5];
ry(0.5810160027637852) q[2];
ry(0.30396396108557017) q[5];
cx q[2],q[5];
ry(-2.2040258813260554) q[3];
ry(-0.9814230622371488) q[4];
cx q[3],q[4];
ry(1.4778343377194532) q[3];
ry(-2.77507692299672) q[4];
cx q[3],q[4];
ry(-2.263122585428806) q[4];
ry(-0.6715487001927114) q[7];
cx q[4],q[7];
ry(2.479144996836351) q[4];
ry(-1.9965869438001556) q[7];
cx q[4],q[7];
ry(2.705873021179929) q[5];
ry(-1.9954810379885153) q[6];
cx q[5],q[6];
ry(0.30830624562392844) q[5];
ry(1.4514448591066227) q[6];
cx q[5],q[6];
ry(-0.7205244715986421) q[6];
ry(0.5130342059508379) q[9];
cx q[6],q[9];
ry(1.1543621581347783) q[6];
ry(2.7713339611413867) q[9];
cx q[6],q[9];
ry(0.1658419457164735) q[7];
ry(2.351473101750928) q[8];
cx q[7],q[8];
ry(2.245930081009667) q[7];
ry(-2.555054810438043) q[8];
cx q[7],q[8];
ry(1.1403063856610984) q[8];
ry(-2.457958057152136) q[11];
cx q[8],q[11];
ry(0.683818896740159) q[8];
ry(-2.8575629221778307) q[11];
cx q[8],q[11];
ry(0.296398135272644) q[9];
ry(-1.4106651007117208) q[10];
cx q[9],q[10];
ry(-1.029465819593787) q[9];
ry(1.841563556577404) q[10];
cx q[9],q[10];
ry(-0.07878302872570586) q[0];
ry(0.9961381961563552) q[1];
cx q[0],q[1];
ry(-1.044497107622773) q[0];
ry(1.742800652529759) q[1];
cx q[0],q[1];
ry(-2.454540664545369) q[2];
ry(0.960386823501092) q[3];
cx q[2],q[3];
ry(-2.4402123946706387) q[2];
ry(2.5003030024477124) q[3];
cx q[2],q[3];
ry(-2.0195836660084607) q[4];
ry(1.4952704433340518) q[5];
cx q[4],q[5];
ry(1.304084683064065) q[4];
ry(-0.6521641517346124) q[5];
cx q[4],q[5];
ry(-1.4891889997513699) q[6];
ry(-1.084463333633885) q[7];
cx q[6],q[7];
ry(-0.7733128010218565) q[6];
ry(-1.263144141744105) q[7];
cx q[6],q[7];
ry(1.9821273470941163) q[8];
ry(2.9999441192280196) q[9];
cx q[8],q[9];
ry(-2.4003264178497234) q[8];
ry(1.1235343477239668) q[9];
cx q[8],q[9];
ry(-1.022506253699384) q[10];
ry(-1.0401576726618744) q[11];
cx q[10],q[11];
ry(2.8077570250262798) q[10];
ry(0.6345697665251905) q[11];
cx q[10],q[11];
ry(-1.2358906134309144) q[0];
ry(-2.635885394429536) q[2];
cx q[0],q[2];
ry(0.48988218214611634) q[0];
ry(-1.4698424332058515) q[2];
cx q[0],q[2];
ry(-2.31880616816514) q[2];
ry(-3.0953939078647155) q[4];
cx q[2],q[4];
ry(-0.33844389456100865) q[2];
ry(1.3695481992092873) q[4];
cx q[2],q[4];
ry(-1.093302880051886) q[4];
ry(0.47626213505429665) q[6];
cx q[4],q[6];
ry(2.6861032572796564) q[4];
ry(-2.1137423979813565) q[6];
cx q[4],q[6];
ry(1.0204490106794717) q[6];
ry(2.1964189173979562) q[8];
cx q[6],q[8];
ry(-2.748005295299038) q[6];
ry(-1.1084168231077391) q[8];
cx q[6],q[8];
ry(-3.046084976433481) q[8];
ry(-0.9840052623787017) q[10];
cx q[8],q[10];
ry(1.9745145519982288) q[8];
ry(2.7246637389818305) q[10];
cx q[8],q[10];
ry(-0.33405162365764696) q[1];
ry(-2.2100036159279237) q[3];
cx q[1],q[3];
ry(-1.3569679345897574) q[1];
ry(-1.7334280056110432) q[3];
cx q[1],q[3];
ry(-0.5337307239240667) q[3];
ry(-0.24032255062581598) q[5];
cx q[3],q[5];
ry(0.10755371964266036) q[3];
ry(-2.252615482936793) q[5];
cx q[3],q[5];
ry(1.8869287697555464) q[5];
ry(-1.4915579228596203) q[7];
cx q[5],q[7];
ry(-0.8169698191801267) q[5];
ry(-2.0615120524867705) q[7];
cx q[5],q[7];
ry(2.9989700544473137) q[7];
ry(1.5098282719523493) q[9];
cx q[7],q[9];
ry(2.655861143733716) q[7];
ry(0.3443418406896876) q[9];
cx q[7],q[9];
ry(0.5760727807051297) q[9];
ry(-1.2576356913329594) q[11];
cx q[9],q[11];
ry(-0.9943004741561544) q[9];
ry(-0.3704429221574283) q[11];
cx q[9],q[11];
ry(0.6644982325369204) q[0];
ry(1.6341434765984624) q[3];
cx q[0],q[3];
ry(2.9811803729687725) q[0];
ry(0.2790477106887029) q[3];
cx q[0],q[3];
ry(2.960375232856093) q[1];
ry(-0.9513044873663777) q[2];
cx q[1],q[2];
ry(0.5892792233445636) q[1];
ry(-1.5647357965337245) q[2];
cx q[1],q[2];
ry(0.9764621325595337) q[2];
ry(1.7306067696791017) q[5];
cx q[2],q[5];
ry(-2.1625975168941567) q[2];
ry(1.239432631015788) q[5];
cx q[2],q[5];
ry(-2.2481559221103113) q[3];
ry(1.5071003604407036) q[4];
cx q[3],q[4];
ry(-0.6365132684984393) q[3];
ry(2.4516486502451627) q[4];
cx q[3],q[4];
ry(2.955080849405099) q[4];
ry(3.0034189840455814) q[7];
cx q[4],q[7];
ry(0.22535610496504377) q[4];
ry(2.5712165270787435) q[7];
cx q[4],q[7];
ry(1.9051717634957024) q[5];
ry(-0.4725271991679634) q[6];
cx q[5],q[6];
ry(0.8739956832418941) q[5];
ry(-2.9586598338378263) q[6];
cx q[5],q[6];
ry(2.2579457790189) q[6];
ry(-2.495148003312526) q[9];
cx q[6],q[9];
ry(-1.5924200117372413) q[6];
ry(-0.6423609090491551) q[9];
cx q[6],q[9];
ry(-1.5816855181137461) q[7];
ry(-1.8746828979676122) q[8];
cx q[7],q[8];
ry(0.3160817200642825) q[7];
ry(3.0602991568273246) q[8];
cx q[7],q[8];
ry(-2.714597331105257) q[8];
ry(2.8513090311094276) q[11];
cx q[8],q[11];
ry(1.0412825044461334) q[8];
ry(2.6444366260988192) q[11];
cx q[8],q[11];
ry(2.66336485363866) q[9];
ry(-0.07014534841852095) q[10];
cx q[9],q[10];
ry(-2.018105643865953) q[9];
ry(2.246423635151894) q[10];
cx q[9],q[10];
ry(1.7311261381475278) q[0];
ry(0.15614374424432495) q[1];
cx q[0],q[1];
ry(1.968528528777263) q[0];
ry(2.245644943044414) q[1];
cx q[0],q[1];
ry(-0.012141362951472168) q[2];
ry(2.3449921146713346) q[3];
cx q[2],q[3];
ry(0.5919013952610899) q[2];
ry(-2.1557846553600797) q[3];
cx q[2],q[3];
ry(-2.7013977017466053) q[4];
ry(-2.2521979278969697) q[5];
cx q[4],q[5];
ry(0.6958725030592658) q[4];
ry(2.4610232387990254) q[5];
cx q[4],q[5];
ry(-0.1589411013100861) q[6];
ry(0.7876826856303998) q[7];
cx q[6],q[7];
ry(2.3359295543458907) q[6];
ry(1.0638464917889299) q[7];
cx q[6],q[7];
ry(-1.7462041403451454) q[8];
ry(0.2614503272949452) q[9];
cx q[8],q[9];
ry(0.2201142392306532) q[8];
ry(-2.145391109144948) q[9];
cx q[8],q[9];
ry(-0.058833538241756855) q[10];
ry(1.9987190895538025) q[11];
cx q[10],q[11];
ry(-0.758457434688703) q[10];
ry(1.2120697815588215) q[11];
cx q[10],q[11];
ry(-1.7761052957530636) q[0];
ry(0.27834206865879735) q[2];
cx q[0],q[2];
ry(-1.044094249337508) q[0];
ry(0.947287634790257) q[2];
cx q[0],q[2];
ry(-0.9137724396035083) q[2];
ry(0.1232162657695529) q[4];
cx q[2],q[4];
ry(1.7544827157930207) q[2];
ry(1.320082149064295) q[4];
cx q[2],q[4];
ry(0.23617352181919582) q[4];
ry(0.7593666221990053) q[6];
cx q[4],q[6];
ry(-2.33413300264945) q[4];
ry(-1.4800149944445664) q[6];
cx q[4],q[6];
ry(-1.0957285106655128) q[6];
ry(1.4615429852847495) q[8];
cx q[6],q[8];
ry(-0.5974644208391169) q[6];
ry(1.3197112301118525) q[8];
cx q[6],q[8];
ry(0.6050634448258188) q[8];
ry(0.8802084494596318) q[10];
cx q[8],q[10];
ry(2.8722114802718726) q[8];
ry(-2.558918357470709) q[10];
cx q[8],q[10];
ry(-1.1251968383407758) q[1];
ry(0.20141526553792222) q[3];
cx q[1],q[3];
ry(1.8131510304892657) q[1];
ry(1.590115702994421) q[3];
cx q[1],q[3];
ry(0.48082041007265275) q[3];
ry(1.15327296450939) q[5];
cx q[3],q[5];
ry(0.4513735850866374) q[3];
ry(1.5028637652298977) q[5];
cx q[3],q[5];
ry(2.1765691489535937) q[5];
ry(-0.6304982313369296) q[7];
cx q[5],q[7];
ry(2.689524211998081) q[5];
ry(-0.4040745784025514) q[7];
cx q[5],q[7];
ry(2.556610671137203) q[7];
ry(0.7427690247880463) q[9];
cx q[7],q[9];
ry(-0.7541612979935719) q[7];
ry(2.0854752293424434) q[9];
cx q[7],q[9];
ry(2.731074615123636) q[9];
ry(2.3922984345687506) q[11];
cx q[9],q[11];
ry(1.4219166095657487) q[9];
ry(0.7512453779246063) q[11];
cx q[9],q[11];
ry(-2.7583873423398924) q[0];
ry(-0.024916886760895096) q[3];
cx q[0],q[3];
ry(-2.1669916219974494) q[0];
ry(-0.77330422149159) q[3];
cx q[0],q[3];
ry(-2.3777083184158614) q[1];
ry(-0.737170748666677) q[2];
cx q[1],q[2];
ry(1.5076788577086493) q[1];
ry(1.018218263888497) q[2];
cx q[1],q[2];
ry(0.8459298852529518) q[2];
ry(2.5338353554874966) q[5];
cx q[2],q[5];
ry(-0.6052129745973004) q[2];
ry(-0.22378514563783636) q[5];
cx q[2],q[5];
ry(1.4421864854628774) q[3];
ry(3.063836181181748) q[4];
cx q[3],q[4];
ry(-1.2772335504207897) q[3];
ry(-0.5977272395633484) q[4];
cx q[3],q[4];
ry(-2.964564593960367) q[4];
ry(1.4150172686958584) q[7];
cx q[4],q[7];
ry(2.915026726404898) q[4];
ry(-2.818457975826244) q[7];
cx q[4],q[7];
ry(-0.7084538124622577) q[5];
ry(-0.7327211053586904) q[6];
cx q[5],q[6];
ry(2.6962462086245096) q[5];
ry(-0.3726737293457341) q[6];
cx q[5],q[6];
ry(1.3847234958177956) q[6];
ry(-0.7681482891230997) q[9];
cx q[6],q[9];
ry(-0.7523794888650361) q[6];
ry(2.60036136282891) q[9];
cx q[6],q[9];
ry(0.9621526937321738) q[7];
ry(-0.4918198218306422) q[8];
cx q[7],q[8];
ry(0.4563378893582061) q[7];
ry(-0.8461868000440165) q[8];
cx q[7],q[8];
ry(0.04046922988031909) q[8];
ry(-1.8584172077797465) q[11];
cx q[8],q[11];
ry(-1.4983286220136396) q[8];
ry(0.44567717627001535) q[11];
cx q[8],q[11];
ry(-2.78103691211523) q[9];
ry(-2.8603558994954374) q[10];
cx q[9],q[10];
ry(2.6744442439670655) q[9];
ry(0.9830404903751733) q[10];
cx q[9],q[10];
ry(-1.9480356341378648) q[0];
ry(-2.8514830958518873) q[1];
cx q[0],q[1];
ry(-2.001227135174376) q[0];
ry(-0.4088897325123882) q[1];
cx q[0],q[1];
ry(3.1370646569157112) q[2];
ry(0.5842548480654264) q[3];
cx q[2],q[3];
ry(1.954577531508291) q[2];
ry(-2.454178932454338) q[3];
cx q[2],q[3];
ry(0.5422129562812088) q[4];
ry(1.2165018723122394) q[5];
cx q[4],q[5];
ry(2.6135604292435213) q[4];
ry(0.8262060359716857) q[5];
cx q[4],q[5];
ry(0.665521076513842) q[6];
ry(-1.5657142289755581) q[7];
cx q[6],q[7];
ry(2.109453524105747) q[6];
ry(1.4559606100460556) q[7];
cx q[6],q[7];
ry(0.5736298056966237) q[8];
ry(-0.585613230336695) q[9];
cx q[8],q[9];
ry(1.3394479207155694) q[8];
ry(0.6107326133431505) q[9];
cx q[8],q[9];
ry(-1.338197097089072) q[10];
ry(0.9345539769801148) q[11];
cx q[10],q[11];
ry(0.8660738853994108) q[10];
ry(2.362962974170472) q[11];
cx q[10],q[11];
ry(0.7637931380679619) q[0];
ry(1.9956446354514572) q[2];
cx q[0],q[2];
ry(-2.468185672759417) q[0];
ry(-1.0630586166414286) q[2];
cx q[0],q[2];
ry(0.8308496146966364) q[2];
ry(-2.850035101185225) q[4];
cx q[2],q[4];
ry(2.3188921026847784) q[2];
ry(0.8586226647556607) q[4];
cx q[2],q[4];
ry(-0.3265884369532266) q[4];
ry(2.3672234495348454) q[6];
cx q[4],q[6];
ry(0.5261513969370659) q[4];
ry(0.3549938043370048) q[6];
cx q[4],q[6];
ry(1.1220374778827182) q[6];
ry(1.8439272742750088) q[8];
cx q[6],q[8];
ry(1.6636243275975306) q[6];
ry(0.2361650743039003) q[8];
cx q[6],q[8];
ry(-1.729206510351454) q[8];
ry(1.523361427814232) q[10];
cx q[8],q[10];
ry(2.764310697351432) q[8];
ry(-2.4904568471411053) q[10];
cx q[8],q[10];
ry(0.418540174657436) q[1];
ry(-0.4770347038581554) q[3];
cx q[1],q[3];
ry(1.8998663359002947) q[1];
ry(1.539254302378033) q[3];
cx q[1],q[3];
ry(-3.0078360102780657) q[3];
ry(-0.9696801915140574) q[5];
cx q[3],q[5];
ry(0.4783489224469801) q[3];
ry(0.7640939581327721) q[5];
cx q[3],q[5];
ry(1.7103240437500353) q[5];
ry(-2.8626977306988666) q[7];
cx q[5],q[7];
ry(2.108913530380378) q[5];
ry(-0.40646260035384696) q[7];
cx q[5],q[7];
ry(1.8948617042649225) q[7];
ry(-0.3624765027070578) q[9];
cx q[7],q[9];
ry(2.326033767081444) q[7];
ry(-0.39315788910078187) q[9];
cx q[7],q[9];
ry(1.9666075716766906) q[9];
ry(-1.861168377731709) q[11];
cx q[9],q[11];
ry(-1.8099624239701217) q[9];
ry(-1.8878549320355946) q[11];
cx q[9],q[11];
ry(2.7279537854157) q[0];
ry(-1.837055659975996) q[3];
cx q[0],q[3];
ry(1.5662498192469771) q[0];
ry(1.6641076557233714) q[3];
cx q[0],q[3];
ry(-0.9680857471507994) q[1];
ry(-0.4649235632634998) q[2];
cx q[1],q[2];
ry(-2.617064365417373) q[1];
ry(-1.5883807549784283) q[2];
cx q[1],q[2];
ry(0.13067695631147028) q[2];
ry(-2.9109833442235313) q[5];
cx q[2],q[5];
ry(-2.1306742808231487) q[2];
ry(1.9833368013721282) q[5];
cx q[2],q[5];
ry(-2.1843029369521174) q[3];
ry(0.918817224517336) q[4];
cx q[3],q[4];
ry(-2.436202383037103) q[3];
ry(0.3402547372233962) q[4];
cx q[3],q[4];
ry(-0.9181697639757649) q[4];
ry(0.8438179224223029) q[7];
cx q[4],q[7];
ry(1.7036443378324366) q[4];
ry(1.3361528976508419) q[7];
cx q[4],q[7];
ry(1.6874439723504304) q[5];
ry(-3.088814825351591) q[6];
cx q[5],q[6];
ry(2.137713201083752) q[5];
ry(-0.856923755631863) q[6];
cx q[5],q[6];
ry(-0.937151261432235) q[6];
ry(2.154015726962701) q[9];
cx q[6],q[9];
ry(-1.622524692998914) q[6];
ry(-1.8677955253344038) q[9];
cx q[6],q[9];
ry(1.001207525133782) q[7];
ry(1.426148840116752) q[8];
cx q[7],q[8];
ry(-0.6072126339334935) q[7];
ry(-0.3397827056491609) q[8];
cx q[7],q[8];
ry(-0.6049182257518018) q[8];
ry(-1.1566856658667284) q[11];
cx q[8],q[11];
ry(2.8141999946055365) q[8];
ry(-2.4573285910787224) q[11];
cx q[8],q[11];
ry(-1.6077522119301544) q[9];
ry(-1.8659948573353926) q[10];
cx q[9],q[10];
ry(-3.0070298713434727) q[9];
ry(1.9655340099247) q[10];
cx q[9],q[10];
ry(-2.1978968270467547) q[0];
ry(-2.9717454785060626) q[1];
cx q[0],q[1];
ry(-0.71685646287435) q[0];
ry(-2.8500814626701825) q[1];
cx q[0],q[1];
ry(0.21463460625260478) q[2];
ry(1.2016618249771955) q[3];
cx q[2],q[3];
ry(-2.1555846801433827) q[2];
ry(-2.122172057534881) q[3];
cx q[2],q[3];
ry(1.7258742030715046) q[4];
ry(-3.0182458397505676) q[5];
cx q[4],q[5];
ry(1.5157293940299974) q[4];
ry(1.2224911119550324) q[5];
cx q[4],q[5];
ry(-2.867411644635285) q[6];
ry(0.1788319841763668) q[7];
cx q[6],q[7];
ry(2.777300766381901) q[6];
ry(1.5134305597696325) q[7];
cx q[6],q[7];
ry(1.4110163420824406) q[8];
ry(-0.9116826707220775) q[9];
cx q[8],q[9];
ry(1.1215514761675691) q[8];
ry(2.499706608969492) q[9];
cx q[8],q[9];
ry(-0.8382365714393245) q[10];
ry(3.059816562446221) q[11];
cx q[10],q[11];
ry(0.6516467403143809) q[10];
ry(1.1594551442328491) q[11];
cx q[10],q[11];
ry(2.032637736085454) q[0];
ry(-1.7576459060529048) q[2];
cx q[0],q[2];
ry(1.578684728461577) q[0];
ry(-0.948215722879074) q[2];
cx q[0],q[2];
ry(-2.5396553320308835) q[2];
ry(-0.9348584789242836) q[4];
cx q[2],q[4];
ry(0.5172362549603869) q[2];
ry(2.6277642168539073) q[4];
cx q[2],q[4];
ry(-1.565366635634983) q[4];
ry(-1.1583184573447562) q[6];
cx q[4],q[6];
ry(-1.3043548118370616) q[4];
ry(0.3115224659983218) q[6];
cx q[4],q[6];
ry(-0.4315919707631961) q[6];
ry(-0.8068124434725288) q[8];
cx q[6],q[8];
ry(-1.6253443647329142) q[6];
ry(-0.8069879465668865) q[8];
cx q[6],q[8];
ry(-2.783671426552451) q[8];
ry(0.8751856645192061) q[10];
cx q[8],q[10];
ry(-1.9401223639077534) q[8];
ry(2.3635561589346685) q[10];
cx q[8],q[10];
ry(2.4321961854869376) q[1];
ry(0.6985035866521988) q[3];
cx q[1],q[3];
ry(0.8301897121320554) q[1];
ry(-0.710011751018444) q[3];
cx q[1],q[3];
ry(1.4098753844983574) q[3];
ry(-1.7023566039501237) q[5];
cx q[3],q[5];
ry(-2.2834965866666708) q[3];
ry(-2.747315807004472) q[5];
cx q[3],q[5];
ry(2.419529734044762) q[5];
ry(2.1171112386018116) q[7];
cx q[5],q[7];
ry(1.5502696967226444) q[5];
ry(0.43783194333445247) q[7];
cx q[5],q[7];
ry(2.3138969162952585) q[7];
ry(-1.858511286584188) q[9];
cx q[7],q[9];
ry(-2.9210628238885943) q[7];
ry(-2.642820392462646) q[9];
cx q[7],q[9];
ry(0.7704375366292221) q[9];
ry(-2.689105421614679) q[11];
cx q[9],q[11];
ry(-1.9051012135211254) q[9];
ry(-2.5357810830971204) q[11];
cx q[9],q[11];
ry(2.9769262085955592) q[0];
ry(-2.3451932758901854) q[3];
cx q[0],q[3];
ry(-2.0987029873520298) q[0];
ry(-0.46093760506822434) q[3];
cx q[0],q[3];
ry(-3.0528373758840375) q[1];
ry(-1.5005417199850144) q[2];
cx q[1],q[2];
ry(-2.024816225146727) q[1];
ry(0.27339946611186844) q[2];
cx q[1],q[2];
ry(2.951280755926893) q[2];
ry(2.2131375097621118) q[5];
cx q[2],q[5];
ry(-0.4084632376796834) q[2];
ry(-1.26828847397113) q[5];
cx q[2],q[5];
ry(-0.23072954587933392) q[3];
ry(0.9059166736151499) q[4];
cx q[3],q[4];
ry(1.0303614046875231) q[3];
ry(-2.6237152619785986) q[4];
cx q[3],q[4];
ry(2.4579086643055477) q[4];
ry(2.540824822002942) q[7];
cx q[4],q[7];
ry(-0.29033868726825085) q[4];
ry(0.2904822293836524) q[7];
cx q[4],q[7];
ry(-2.0379853004486295) q[5];
ry(0.5079477076431916) q[6];
cx q[5],q[6];
ry(1.212918578906344) q[5];
ry(-1.6345181481698834) q[6];
cx q[5],q[6];
ry(-2.803959004027674) q[6];
ry(-1.0282011305019614) q[9];
cx q[6],q[9];
ry(-2.4289410767842) q[6];
ry(0.9659930016003959) q[9];
cx q[6],q[9];
ry(0.013475870877369124) q[7];
ry(-2.7036095364543278) q[8];
cx q[7],q[8];
ry(-1.4187502518310977) q[7];
ry(1.9217662578980577) q[8];
cx q[7],q[8];
ry(-3.016427549996796) q[8];
ry(-2.630095258579313) q[11];
cx q[8],q[11];
ry(2.1892054999977164) q[8];
ry(1.3883947734007256) q[11];
cx q[8],q[11];
ry(0.048938802985058594) q[9];
ry(-0.7603025309186383) q[10];
cx q[9],q[10];
ry(1.3579664967326173) q[9];
ry(-3.0112037595895025) q[10];
cx q[9],q[10];
ry(-1.741288253454367) q[0];
ry(3.0364482021646952) q[1];
ry(-2.445798365321272) q[2];
ry(-0.8373964307390245) q[3];
ry(-0.4708543293198396) q[4];
ry(-2.653862066929929) q[5];
ry(-1.0584989879299274) q[6];
ry(-3.094166071944121) q[7];
ry(-1.647548771899003) q[8];
ry(-1.0729345902052003) q[9];
ry(-1.769738623262278) q[10];
ry(-2.754911112614972) q[11];