OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.4539910643342484) q[0];
ry(0.8756539780130668) q[1];
cx q[0],q[1];
ry(1.1041752771153979) q[0];
ry(1.1723720843940861) q[1];
cx q[0],q[1];
ry(2.266496246285646) q[1];
ry(0.09779050111116873) q[2];
cx q[1],q[2];
ry(-0.1315547273092541) q[1];
ry(1.7608844852425696) q[2];
cx q[1],q[2];
ry(-0.10885462642339018) q[2];
ry(0.5086365289914507) q[3];
cx q[2],q[3];
ry(-1.9891756997722139) q[2];
ry(-0.35141205227486344) q[3];
cx q[2],q[3];
ry(0.9983035954155364) q[3];
ry(-0.7087574229381823) q[4];
cx q[3],q[4];
ry(0.4195809379222908) q[3];
ry(1.4988793028465044) q[4];
cx q[3],q[4];
ry(-0.8368823044121304) q[4];
ry(1.7760070003705106) q[5];
cx q[4],q[5];
ry(-1.1864518774347568) q[4];
ry(-2.4365460003775774) q[5];
cx q[4],q[5];
ry(-2.397774551235201) q[5];
ry(-2.7336804790400606) q[6];
cx q[5],q[6];
ry(-2.7492458967016207) q[5];
ry(2.1764987692163076) q[6];
cx q[5],q[6];
ry(2.8361777554116396) q[6];
ry(-0.21800939823430115) q[7];
cx q[6],q[7];
ry(-1.2196654222314276) q[6];
ry(-1.9406755168504457) q[7];
cx q[6],q[7];
ry(1.3575795457068405) q[7];
ry(0.0755683667650926) q[8];
cx q[7],q[8];
ry(0.21580549769385193) q[7];
ry(-0.7079555158743259) q[8];
cx q[7],q[8];
ry(-1.8832300934274477) q[8];
ry(-0.5255515253293706) q[9];
cx q[8],q[9];
ry(1.2359713024929746) q[8];
ry(1.8451923403542767) q[9];
cx q[8],q[9];
ry(0.550868095446382) q[9];
ry(-0.9309999096730008) q[10];
cx q[9],q[10];
ry(-1.3916382870536426) q[9];
ry(-2.0159982105959076) q[10];
cx q[9],q[10];
ry(-1.3090863588478086) q[10];
ry(-2.4065214482850164) q[11];
cx q[10],q[11];
ry(1.8019421934220918) q[10];
ry(-2.1742518677160816) q[11];
cx q[10],q[11];
ry(2.080858969771148) q[11];
ry(0.5422382834658457) q[12];
cx q[11],q[12];
ry(0.049199383448941134) q[11];
ry(-1.0197317305180498) q[12];
cx q[11],q[12];
ry(0.8139268832646702) q[12];
ry(-2.9321087149466445) q[13];
cx q[12],q[13];
ry(0.5676304224501827) q[12];
ry(0.9179504758569681) q[13];
cx q[12],q[13];
ry(2.593165559874008) q[13];
ry(-1.6569020065309763) q[14];
cx q[13],q[14];
ry(-1.4731629937275874) q[13];
ry(0.9447857509933794) q[14];
cx q[13],q[14];
ry(0.9120599259724816) q[14];
ry(-2.518492738957856) q[15];
cx q[14],q[15];
ry(1.3960401036454098) q[14];
ry(0.9205543232051717) q[15];
cx q[14],q[15];
ry(2.365996045874933) q[15];
ry(1.2928511163988166) q[16];
cx q[15],q[16];
ry(1.96697879814527) q[15];
ry(-2.541482031657477) q[16];
cx q[15],q[16];
ry(-0.611497013777999) q[16];
ry(1.8271122248906835) q[17];
cx q[16],q[17];
ry(2.226762405896353) q[16];
ry(-3.0063620583025217) q[17];
cx q[16],q[17];
ry(0.5583967488354245) q[17];
ry(-2.868731667260064) q[18];
cx q[17],q[18];
ry(2.3620218125170243) q[17];
ry(2.0675526687135184) q[18];
cx q[17],q[18];
ry(-2.237564164952622) q[18];
ry(1.5311773092852743) q[19];
cx q[18],q[19];
ry(2.517175627765769) q[18];
ry(3.137982795490033) q[19];
cx q[18],q[19];
ry(0.616948774341644) q[0];
ry(-0.8303152642504026) q[1];
cx q[0],q[1];
ry(1.9948576406110918) q[0];
ry(-0.01206664437375509) q[1];
cx q[0],q[1];
ry(2.5352422286243184) q[1];
ry(-3.0760448040287334) q[2];
cx q[1],q[2];
ry(-0.17413871844028436) q[1];
ry(2.9863000154940895) q[2];
cx q[1],q[2];
ry(-2.9678600794427132) q[2];
ry(0.7249737931948594) q[3];
cx q[2],q[3];
ry(-2.022458808961709) q[2];
ry(0.5269041079487069) q[3];
cx q[2],q[3];
ry(-2.952436051910588) q[3];
ry(1.4518527105518704) q[4];
cx q[3],q[4];
ry(-0.19711871335920256) q[3];
ry(-2.3718489308885076) q[4];
cx q[3],q[4];
ry(1.9635070686468408) q[4];
ry(0.6233048534864185) q[5];
cx q[4],q[5];
ry(0.0053162751841426825) q[4];
ry(0.1520621359673786) q[5];
cx q[4],q[5];
ry(-1.0354901752056094) q[5];
ry(-2.168614960819301) q[6];
cx q[5],q[6];
ry(2.67804608971568) q[5];
ry(-0.6449363513187731) q[6];
cx q[5],q[6];
ry(1.2138739995977674) q[6];
ry(-1.4704351167134435) q[7];
cx q[6],q[7];
ry(2.42902083953134) q[6];
ry(1.5555089537127318) q[7];
cx q[6],q[7];
ry(1.9849109278089034) q[7];
ry(1.4482383120186224) q[8];
cx q[7],q[8];
ry(-2.9153904592966406) q[7];
ry(0.22368308066256434) q[8];
cx q[7],q[8];
ry(2.4320820261364737) q[8];
ry(2.039362289393214) q[9];
cx q[8],q[9];
ry(0.178933370980319) q[8];
ry(2.9414134604491347) q[9];
cx q[8],q[9];
ry(-0.8641389696289677) q[9];
ry(-3.1066397270802657) q[10];
cx q[9],q[10];
ry(-0.5464469956714959) q[9];
ry(2.7606644869977726) q[10];
cx q[9],q[10];
ry(-0.13469573794786588) q[10];
ry(1.6708924205117908) q[11];
cx q[10],q[11];
ry(-0.5484101078021384) q[10];
ry(0.7140239696214897) q[11];
cx q[10],q[11];
ry(0.6532081781184413) q[11];
ry(1.2554025550499084) q[12];
cx q[11],q[12];
ry(2.7791034339773466) q[11];
ry(-0.0944493570269449) q[12];
cx q[11],q[12];
ry(1.9464333811868533) q[12];
ry(-0.8796766166348524) q[13];
cx q[12],q[13];
ry(2.5040252941759777) q[12];
ry(-3.1082256147005265) q[13];
cx q[12],q[13];
ry(-1.8273643672406061) q[13];
ry(-1.5383850490536528) q[14];
cx q[13],q[14];
ry(0.07881115821550286) q[13];
ry(3.111677370883143) q[14];
cx q[13],q[14];
ry(-1.7971156227838394) q[14];
ry(-2.3587298218838306) q[15];
cx q[14],q[15];
ry(-2.8388971829215297) q[14];
ry(-1.8157004807334127) q[15];
cx q[14],q[15];
ry(-1.3941703734757525) q[15];
ry(2.6972799707697597) q[16];
cx q[15],q[16];
ry(-3.0417525505238063) q[15];
ry(2.965128939624628) q[16];
cx q[15],q[16];
ry(-2.348691972525616) q[16];
ry(-0.9320506875250238) q[17];
cx q[16],q[17];
ry(1.8935433314963905) q[16];
ry(1.1323313971458004) q[17];
cx q[16],q[17];
ry(-1.3487529891303167) q[17];
ry(-2.7839902676867503) q[18];
cx q[17],q[18];
ry(-0.45490617204394973) q[17];
ry(-2.566119905831719) q[18];
cx q[17],q[18];
ry(-1.4238538793634061) q[18];
ry(-2.329799873274108) q[19];
cx q[18],q[19];
ry(-2.7859423376126) q[18];
ry(-0.31369409594404635) q[19];
cx q[18],q[19];
ry(-1.851666193388871) q[0];
ry(-1.8743569872836794) q[1];
cx q[0],q[1];
ry(2.204968367614566) q[0];
ry(-2.607130039239119) q[1];
cx q[0],q[1];
ry(-1.9417909094980002) q[1];
ry(0.7358384416547437) q[2];
cx q[1],q[2];
ry(2.2166285099122502) q[1];
ry(0.6637487526074608) q[2];
cx q[1],q[2];
ry(1.4349856423247527) q[2];
ry(-1.5679345110826897) q[3];
cx q[2],q[3];
ry(-1.2773195309652312) q[2];
ry(-3.044736574379694) q[3];
cx q[2],q[3];
ry(1.7299317010721738) q[3];
ry(1.060243741401897) q[4];
cx q[3],q[4];
ry(-1.84118521192772) q[3];
ry(0.6339625935433341) q[4];
cx q[3],q[4];
ry(-0.19738424796848497) q[4];
ry(1.473506694516848) q[5];
cx q[4],q[5];
ry(-1.5299898882698038) q[4];
ry(0.18963474233571453) q[5];
cx q[4],q[5];
ry(-2.9382673435403412) q[5];
ry(1.4551517840158308) q[6];
cx q[5],q[6];
ry(-1.5347319911812072) q[5];
ry(0.0033732562820791836) q[6];
cx q[5],q[6];
ry(-1.9306809794498099) q[6];
ry(-1.9327707838735428) q[7];
cx q[6],q[7];
ry(1.5548570079434647) q[6];
ry(1.4411621307298024) q[7];
cx q[6],q[7];
ry(-0.010533175009658889) q[7];
ry(2.710231594900729) q[8];
cx q[7],q[8];
ry(-1.5555728996242646) q[7];
ry(-2.8469602543199075) q[8];
cx q[7],q[8];
ry(-1.8151269472375937) q[8];
ry(0.4972895496383609) q[9];
cx q[8],q[9];
ry(1.7450419631207337) q[8];
ry(-0.02740039307751996) q[9];
cx q[8],q[9];
ry(1.5340621326115438) q[9];
ry(1.6200885467594537) q[10];
cx q[9],q[10];
ry(0.24360965529479994) q[9];
ry(-2.998740150227789) q[10];
cx q[9],q[10];
ry(-1.5556321014087366) q[10];
ry(2.1923104480183877) q[11];
cx q[10],q[11];
ry(-0.17502402928900998) q[10];
ry(-2.041175901351857) q[11];
cx q[10],q[11];
ry(1.7815718153338118) q[11];
ry(-3.0799817116589088) q[12];
cx q[11],q[12];
ry(-0.053182013428664916) q[11];
ry(0.5924655563467243) q[12];
cx q[11],q[12];
ry(-2.75401633559788) q[12];
ry(-0.06528016506987377) q[13];
cx q[12],q[13];
ry(-0.5804207387510125) q[12];
ry(1.1479901380167012) q[13];
cx q[12],q[13];
ry(2.9604294996099356) q[13];
ry(-1.1815876334837783) q[14];
cx q[13],q[14];
ry(2.9196433223319045) q[13];
ry(2.9105884363305385) q[14];
cx q[13],q[14];
ry(1.3784256983412395) q[14];
ry(1.8756966092941116) q[15];
cx q[14],q[15];
ry(-1.3333357119592835) q[14];
ry(-1.0811188916772183) q[15];
cx q[14],q[15];
ry(-1.2264589952170557) q[15];
ry(2.8759687210370983) q[16];
cx q[15],q[16];
ry(-2.876489367440419) q[15];
ry(-0.12884168851581815) q[16];
cx q[15],q[16];
ry(-2.2291168953194465) q[16];
ry(-0.29294810693741535) q[17];
cx q[16],q[17];
ry(1.51790366201097) q[16];
ry(2.503138446798177) q[17];
cx q[16],q[17];
ry(-1.7879622836189086) q[17];
ry(-2.954272692471187) q[18];
cx q[17],q[18];
ry(-2.787604858934127) q[17];
ry(-2.8824886455140986) q[18];
cx q[17],q[18];
ry(2.7324858113051427) q[18];
ry(-2.4099465026909166) q[19];
cx q[18],q[19];
ry(-1.7901715740388298) q[18];
ry(-2.7690110818502505) q[19];
cx q[18],q[19];
ry(2.028571087853348) q[0];
ry(-2.1790383593519707) q[1];
cx q[0],q[1];
ry(2.2546638999363364) q[0];
ry(-2.0220365064855024) q[1];
cx q[0],q[1];
ry(1.759693243290629) q[1];
ry(1.4411936183334522) q[2];
cx q[1],q[2];
ry(1.6787629089149072) q[1];
ry(0.7262091727268462) q[2];
cx q[1],q[2];
ry(-2.2451174229437836) q[2];
ry(-1.973985482449381) q[3];
cx q[2],q[3];
ry(0.5363154599613029) q[2];
ry(0.35649339607224517) q[3];
cx q[2],q[3];
ry(-2.164410465223656) q[3];
ry(2.6670691026472855) q[4];
cx q[3],q[4];
ry(-3.0786414020466566) q[3];
ry(2.8961790998105035) q[4];
cx q[3],q[4];
ry(0.2433254581634063) q[4];
ry(2.8556293637243826) q[5];
cx q[4],q[5];
ry(2.9276739772729554) q[4];
ry(-2.8696316382276263) q[5];
cx q[4],q[5];
ry(-2.2997728477457597) q[5];
ry(-2.492490584237742) q[6];
cx q[5],q[6];
ry(-3.1369839781604463) q[5];
ry(-0.031445440490944776) q[6];
cx q[5],q[6];
ry(-0.9943190334649704) q[6];
ry(1.2261749879826578) q[7];
cx q[6],q[7];
ry(-3.132125069312844) q[6];
ry(2.7655203564812387) q[7];
cx q[6],q[7];
ry(-0.6510619063634041) q[7];
ry(-0.06324182037142284) q[8];
cx q[7],q[8];
ry(3.13439523885702) q[7];
ry(0.7482724115815595) q[8];
cx q[7],q[8];
ry(-0.30030570101091897) q[8];
ry(1.5575904474500126) q[9];
cx q[8],q[9];
ry(2.97617882534862) q[8];
ry(-0.3202175362714446) q[9];
cx q[8],q[9];
ry(1.4258971719358338) q[9];
ry(1.2467669649728448) q[10];
cx q[9],q[10];
ry(1.7872103258576066) q[9];
ry(-3.076386373928902) q[10];
cx q[9],q[10];
ry(1.2996180490732487) q[10];
ry(1.2061859040510723) q[11];
cx q[10],q[11];
ry(-1.5247270905902708) q[10];
ry(-0.0270453681114045) q[11];
cx q[10],q[11];
ry(-0.38389573411242806) q[11];
ry(1.522282438946603) q[12];
cx q[11],q[12];
ry(1.566764376848104) q[11];
ry(3.138176791028951) q[12];
cx q[11],q[12];
ry(-1.5693708776654063) q[12];
ry(-1.0538378211038302) q[13];
cx q[12],q[13];
ry(-1.5684055814238542) q[12];
ry(-1.0087588891856303) q[13];
cx q[12],q[13];
ry(-1.5704010136933508) q[13];
ry(1.8454081317859927) q[14];
cx q[13],q[14];
ry(-1.5737092461385098) q[13];
ry(1.2693544741706733) q[14];
cx q[13],q[14];
ry(-1.5718152669750314) q[14];
ry(-2.288569432381914) q[15];
cx q[14],q[15];
ry(1.5715256333869387) q[14];
ry(1.6475668346106858) q[15];
cx q[14],q[15];
ry(1.5736756761005093) q[15];
ry(0.2517449714406293) q[16];
cx q[15],q[16];
ry(1.571116781103632) q[15];
ry(0.49742301994379773) q[16];
cx q[15],q[16];
ry(-1.571181036119354) q[16];
ry(-2.006110765025012) q[17];
cx q[16],q[17];
ry(-1.5712722118062932) q[16];
ry(-0.9212818737875322) q[17];
cx q[16],q[17];
ry(1.574965907039757) q[17];
ry(-0.6628984561888998) q[18];
cx q[17],q[18];
ry(-1.5725725187945523) q[17];
ry(-2.246249086221318) q[18];
cx q[17],q[18];
ry(-1.576588192152653) q[18];
ry(-2.3865576083825286) q[19];
cx q[18],q[19];
ry(-1.5768732184569396) q[18];
ry(-2.6903721726441803) q[19];
cx q[18],q[19];
ry(3.0756487837690787) q[0];
ry(-1.7756432097194) q[1];
ry(1.9371165057112467) q[2];
ry(-2.4668737904270235) q[3];
ry(0.7763949001877606) q[4];
ry(2.2168783849775284) q[5];
ry(3.1187393665000926) q[6];
ry(1.991389652298868) q[7];
ry(1.5785857411493693) q[8];
ry(1.419441703679846) q[9];
ry(1.7681344997380604) q[10];
ry(2.7462086356532582) q[11];
ry(1.5695357027587897) q[12];
ry(-1.5695788939543576) q[13];
ry(1.5711605501152708) q[14];
ry(1.569352790579337) q[15];
ry(-1.571837554251563) q[16];
ry(-1.5693621894790546) q[17];
ry(1.570009241058309) q[18];
ry(1.5737266111373907) q[19];