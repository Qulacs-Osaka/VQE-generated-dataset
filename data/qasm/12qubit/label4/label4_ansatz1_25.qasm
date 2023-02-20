OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.862717296523184) q[0];
rz(-2.6154089983814677) q[0];
ry(0.5262479603770425) q[1];
rz(-2.883535934322356) q[1];
ry(-2.17788669436975) q[2];
rz(3.0069626890263494) q[2];
ry(1.7450213804350545) q[3];
rz(0.02412673014323996) q[3];
ry(-1.5887355047773024) q[4];
rz(0.7952587840940493) q[4];
ry(-0.002022907886589018) q[5];
rz(2.314264374349374) q[5];
ry(-1.8389375483264996) q[6];
rz(-0.8205929956668746) q[6];
ry(-1.2377635578727784) q[7];
rz(-1.627615142905328) q[7];
ry(0.47091733766703553) q[8];
rz(-2.573662063362563) q[8];
ry(0.5213133738264398) q[9];
rz(1.4365104194149798) q[9];
ry(-2.095566648737441) q[10];
rz(0.9229975906989019) q[10];
ry(2.6820724873034933) q[11];
rz(1.4787355248586354) q[11];
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
ry(0.7036536644351882) q[0];
rz(-2.798970005701841) q[0];
ry(2.1098872985036374) q[1];
rz(1.014469678032525) q[1];
ry(-2.7881064452451736) q[2];
rz(3.072859294726397) q[2];
ry(-2.011747192009069) q[3];
rz(1.83454960669433) q[3];
ry(1.6126577815651741) q[4];
rz(-2.404227898021439) q[4];
ry(0.022644563146240023) q[5];
rz(1.179337769963066) q[5];
ry(1.2326701349013147) q[6];
rz(-1.8758146354432137) q[6];
ry(-0.8994032668489558) q[7];
rz(-0.7775210492287777) q[7];
ry(3.1042062025228963) q[8];
rz(-2.6337515626631287) q[8];
ry(-3.129260933656507) q[9];
rz(-1.5677270857466343) q[9];
ry(-2.95665100238905) q[10];
rz(-1.9593082544856593) q[10];
ry(-0.13152934336155042) q[11];
rz(1.4384774471191824) q[11];
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
ry(0.8972965828148525) q[0];
rz(-2.9648005290202537) q[0];
ry(1.3907108561110615) q[1];
rz(-2.175826385989528) q[1];
ry(0.27674277000964675) q[2];
rz(-1.0212619532213192) q[2];
ry(2.118392502330371) q[3];
rz(-0.5592866603884428) q[3];
ry(-1.2504900183073215) q[4];
rz(0.9058416827952223) q[4];
ry(-3.1348057796711504) q[5];
rz(-0.6335228322175828) q[5];
ry(0.2331630623644685) q[6];
rz(-0.03067288409277175) q[6];
ry(-2.351665136178597) q[7];
rz(1.4340549565567726) q[7];
ry(3.060602617143019) q[8];
rz(0.3853556731599559) q[8];
ry(-0.3141157705961632) q[9];
rz(0.6641291016837786) q[9];
ry(-0.4797400696274591) q[10];
rz(-2.604053861010522) q[10];
ry(2.143717728510544) q[11];
rz(0.5558320910900042) q[11];
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
ry(-0.21095688073768404) q[0];
rz(-2.6728007311066504) q[0];
ry(-1.929258017119857) q[1];
rz(-0.3209392512702203) q[1];
ry(-0.40036708723263853) q[2];
rz(1.5315982073675034) q[2];
ry(-0.1730548462663002) q[3];
rz(-1.5961734753721482) q[3];
ry(1.958875471422699) q[4];
rz(-0.348178924327331) q[4];
ry(0.0372271914311808) q[5];
rz(-1.5777159982992428) q[5];
ry(2.2993491185815564) q[6];
rz(-1.8651466082726096) q[6];
ry(-0.5222597088035306) q[7];
rz(-0.9425971525448169) q[7];
ry(-3.1173962812397686) q[8];
rz(1.244920768871916) q[8];
ry(3.109182409712058) q[9];
rz(0.33187581534881855) q[9];
ry(1.5265450587445768) q[10];
rz(0.748346898096257) q[10];
ry(-0.04204777854434205) q[11];
rz(0.7086018980983972) q[11];
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
ry(2.7105467468246878) q[0];
rz(1.6178847029908248) q[0];
ry(0.12832749503063584) q[1];
rz(-0.6045948630855568) q[1];
ry(1.0652978609119215) q[2];
rz(2.7216675711508675) q[2];
ry(0.5754757442776188) q[3];
rz(0.5179522015989485) q[3];
ry(1.035873864704834) q[4];
rz(2.449862521008852) q[4];
ry(-2.539782395721038) q[5];
rz(1.7160339464224812) q[5];
ry(-2.699646416592241) q[6];
rz(2.574165612160821) q[6];
ry(-0.6232949249869805) q[7];
rz(2.4120438313707275) q[7];
ry(2.833333620817606) q[8];
rz(-0.31287709104089034) q[8];
ry(1.5892451697882006) q[9];
rz(2.9477987967867154) q[9];
ry(-2.8894679806616534) q[10];
rz(-2.124802559958576) q[10];
ry(1.489450262327277) q[11];
rz(-0.3225559137914934) q[11];
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
ry(2.0460586087495454) q[0];
rz(-3.0570130693957966) q[0];
ry(2.1681148270696093) q[1];
rz(1.1579291741381097) q[1];
ry(1.3851869513863457) q[2];
rz(0.65812750775941) q[2];
ry(2.0511224965780372) q[3];
rz(1.825767759153921) q[3];
ry(-1.5860911917625384) q[4];
rz(-1.9337403573088827) q[4];
ry(1.7087900566734258) q[5];
rz(1.5745295624499152) q[5];
ry(-1.8557269383232682) q[6];
rz(1.5576868735465013) q[6];
ry(-2.7915061775650303) q[7];
rz(1.2681631254647612) q[7];
ry(0.02318243670390085) q[8];
rz(-0.34744115341984116) q[8];
ry(0.04925322240707653) q[9];
rz(0.9105991839149103) q[9];
ry(-1.3687399208314925) q[10];
rz(1.8979699070549252) q[10];
ry(-0.5037877962898413) q[11];
rz(2.5858369087721935) q[11];
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
ry(0.5268737492091591) q[0];
rz(-2.718542375362275) q[0];
ry(1.1828043640123975) q[1];
rz(-1.7639895664636402) q[1];
ry(1.5562606946391409) q[2];
rz(0.9081896495218638) q[2];
ry(3.1206065763246644) q[3];
rz(-0.10137857562299768) q[3];
ry(-0.8270254651715189) q[4];
rz(0.4075836273947112) q[4];
ry(-3.0112924537788968) q[5];
rz(-2.6027080433634358) q[5];
ry(3.138977721600642) q[6];
rz(-1.5845060532196602) q[6];
ry(-2.9115980787031996) q[7];
rz(-2.7160249510949175) q[7];
ry(3.020905909510976) q[8];
rz(1.2691943542599038) q[8];
ry(-1.9968573672662213) q[9];
rz(2.7153570262705093) q[9];
ry(2.628281971809258) q[10];
rz(-3.0669050272937888) q[10];
ry(1.9651722602653239) q[11];
rz(-2.8613740882833265) q[11];
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
ry(-1.0293713325160259) q[0];
rz(2.4966309723583366) q[0];
ry(1.4998745886743299) q[1];
rz(2.9532979525021177) q[1];
ry(-1.925370126758677) q[2];
rz(-2.1989254458112715) q[2];
ry(3.082492287867371) q[3];
rz(0.7038011022353688) q[3];
ry(0.005030970756476627) q[4];
rz(-1.8141318672753042) q[4];
ry(0.018524711237879136) q[5];
rz(-1.245835501659972) q[5];
ry(3.1384182368235267) q[6];
rz(-1.5692039556968598) q[6];
ry(2.216302409124504) q[7];
rz(-1.4321113829108052) q[7];
ry(3.1060170736809654) q[8];
rz(1.8774368353372195) q[8];
ry(-0.038854685372316844) q[9];
rz(-1.8710943877689576) q[9];
ry(-1.4355411944849161) q[10];
rz(-1.4012191998696872) q[10];
ry(3.0149224541923734) q[11];
rz(-1.8328077213463052) q[11];
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
ry(-1.7096044533984984) q[0];
rz(-0.5503071475214495) q[0];
ry(-0.2888251627324932) q[1];
rz(-1.2320078894271322) q[1];
ry(-0.8256520947768982) q[2];
rz(1.6809415263826877) q[2];
ry(3.1323823068735046) q[3];
rz(0.3258976154379064) q[3];
ry(-1.8375006587301073) q[4];
rz(-2.1850672895621956) q[4];
ry(1.7218054918463912) q[5];
rz(-0.2256679364875858) q[5];
ry(0.8454854184319738) q[6];
rz(-1.037471775699135) q[6];
ry(-0.05580782821120433) q[7];
rz(1.4088753358925734) q[7];
ry(-3.126524338496198) q[8];
rz(-0.17603452205659145) q[8];
ry(1.2970936968527005) q[9];
rz(-1.0617122822140272) q[9];
ry(2.7335372813124312) q[10];
rz(1.7088420467031777) q[10];
ry(1.428486763276954) q[11];
rz(-1.5059356993045045) q[11];
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
ry(-1.7631631758101038) q[0];
rz(0.390507573706087) q[0];
ry(1.663409551117903) q[1];
rz(0.16805581609031012) q[1];
ry(-1.764704950259146) q[2];
rz(0.3702263401544563) q[2];
ry(0.3161134850067267) q[3];
rz(-0.2157312396970476) q[3];
ry(3.0896972242632983) q[4];
rz(1.7498326632866614) q[4];
ry(0.006780880234255271) q[5];
rz(-2.8351003342343373) q[5];
ry(3.120384508852209) q[6];
rz(0.8362141005182708) q[6];
ry(-1.477160732639736) q[7];
rz(-0.5189287444789515) q[7];
ry(0.04365672060170287) q[8];
rz(-2.7887293814521867) q[8];
ry(-0.565571749499705) q[9];
rz(2.4372206055104986) q[9];
ry(1.7180217671382607) q[10];
rz(1.6253633342815779) q[10];
ry(-1.0231784112043691) q[11];
rz(2.073129735140243) q[11];
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
ry(-0.71706631103636) q[0];
rz(-0.5463967249290402) q[0];
ry(1.737136414374494) q[1];
rz(0.9316486632256149) q[1];
ry(1.8553668863909598) q[2];
rz(1.8834124623622581) q[2];
ry(-0.020005560643798794) q[3];
rz(-1.0951195901558513) q[3];
ry(-1.6305732272058755) q[4];
rz(-2.886096524533389) q[4];
ry(0.24850346515009633) q[5];
rz(1.1027696763704906) q[5];
ry(-2.7768823583131628) q[6];
rz(-2.9531943446935216) q[6];
ry(-3.055191930136714) q[7];
rz(1.476573085331073) q[7];
ry(-1.592394367183638) q[8];
rz(-2.6993976425854425) q[8];
ry(0.9862201155600888) q[9];
rz(-0.44268861636396895) q[9];
ry(2.9470466572306013) q[10];
rz(2.175728566852981) q[10];
ry(-2.6488642564485154) q[11];
rz(-0.7846718369124099) q[11];
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
ry(1.5961467255336053) q[0];
rz(1.5099545757709052) q[0];
ry(1.4944059312794584) q[1];
rz(-0.937257296833362) q[1];
ry(1.9087441664399334) q[2];
rz(0.23126871902281465) q[2];
ry(-1.710652977353269) q[3];
rz(-1.2671375184143008) q[3];
ry(2.8250478901965974) q[4];
rz(-2.272003777198763) q[4];
ry(0.5044444416652311) q[5];
rz(-0.8749989044739924) q[5];
ry(1.5145379682252802) q[6];
rz(-0.22162756653111781) q[6];
ry(3.1366804322273953) q[7];
rz(-2.061743892664233) q[7];
ry(0.017031784986787812) q[8];
rz(-1.2206319744485774) q[8];
ry(3.1316078241962946) q[9];
rz(-0.4882468540768681) q[9];
ry(-2.837193696749062) q[10];
rz(2.4111116931829573) q[10];
ry(-0.3779407382553856) q[11];
rz(-2.0587236860745044) q[11];
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
ry(0.7439117826983218) q[0];
rz(2.1408599945379985) q[0];
ry(0.517658481357941) q[1];
rz(-0.5923881084372544) q[1];
ry(-2.026560252993201) q[2];
rz(-2.0757832239104306) q[2];
ry(3.1350268083528046) q[3];
rz(-2.243302161124051) q[3];
ry(3.095594343862679) q[4];
rz(1.1246437206663125) q[4];
ry(-0.020088966369151615) q[5];
rz(-2.0201554212786315) q[5];
ry(0.014807440042616804) q[6];
rz(-1.8301778502285568) q[6];
ry(-0.003510093922989037) q[7];
rz(1.0519442564675874) q[7];
ry(-0.021394633423186125) q[8];
rz(0.7798382662916152) q[8];
ry(-1.4129942110861007) q[9];
rz(0.37100699583437186) q[9];
ry(-0.1144713296570119) q[10];
rz(0.6743598631294676) q[10];
ry(1.2678899344207624) q[11];
rz(-2.2674583107717803) q[11];
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
ry(2.273907655647502) q[0];
rz(1.9541628171405294) q[0];
ry(1.4354132138847264) q[1];
rz(-1.3133360326157395) q[1];
ry(1.8468040144041546) q[2];
rz(-1.4817994887945156) q[2];
ry(-1.6368289499183692) q[3];
rz(-1.8055965260990856) q[3];
ry(-2.930733364769775) q[4];
rz(-2.55287349016073) q[4];
ry(-0.44277516301448566) q[5];
rz(-2.6765111896519636) q[5];
ry(2.9896936850420364) q[6];
rz(-2.0101297419368533) q[6];
ry(2.405614977625251) q[7];
rz(0.626260811733336) q[7];
ry(-0.6754972628642538) q[8];
rz(-3.1407535472691444) q[8];
ry(1.568349226627141) q[9];
rz(0.1805951890740607) q[9];
ry(-1.4916628621206005) q[10];
rz(-0.7028906692038) q[10];
ry(1.5578694073048949) q[11];
rz(-0.4274934372907445) q[11];
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
ry(-2.4871956149090733) q[0];
rz(3.055546158599472) q[0];
ry(-0.029903529996118852) q[1];
rz(-3.0278132901966215) q[1];
ry(-0.5544033004969391) q[2];
rz(1.403570928282785) q[2];
ry(-0.001421932521074764) q[3];
rz(-0.028588663301254817) q[3];
ry(0.6326924492359529) q[4];
rz(2.5009400226495235) q[4];
ry(-0.0034239030793310476) q[5];
rz(-0.5781839761738468) q[5];
ry(1.4925116750990153) q[6];
rz(-0.030975194944079302) q[6];
ry(-1.5530506463185627) q[7];
rz(2.5591118140521187) q[7];
ry(-0.4321118614679378) q[8];
rz(-0.7901411152684662) q[8];
ry(1.7404801053476273) q[9];
rz(-3.061048095914536) q[9];
ry(1.5759107366363319) q[10];
rz(0.9593017116042569) q[10];
ry(-0.6332564672635103) q[11];
rz(-2.0473029200304778) q[11];
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
ry(2.2402398078018506) q[0];
rz(0.606795606874759) q[0];
ry(1.045534414723563) q[1];
rz(1.5867019402627918) q[1];
ry(-2.2885175597932688) q[2];
rz(1.1255395327031552) q[2];
ry(0.566139537404855) q[3];
rz(-0.8275509978068811) q[3];
ry(-1.9952990576959833) q[4];
rz(-1.6101529239003982) q[4];
ry(0.08419404050636636) q[5];
rz(-0.41430693722930756) q[5];
ry(-0.010309238268297173) q[6];
rz(-0.15673147281564004) q[6];
ry(-0.02195411164993111) q[7];
rz(2.167664710839519) q[7];
ry(3.1400084523506835) q[8];
rz(-0.39621168949594304) q[8];
ry(0.8743735327615285) q[9];
rz(2.4770906778042003) q[9];
ry(0.0070803954095790544) q[10];
rz(0.09726803776356406) q[10];
ry(1.050128662298743) q[11];
rz(-1.0928027334667392) q[11];
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
ry(-1.7945622429538644) q[0];
rz(1.0926574865126364) q[0];
ry(1.3935199837130572) q[1];
rz(1.5957130568732376) q[1];
ry(-2.284579841957667) q[2];
rz(-0.47014612844570264) q[2];
ry(-3.1376992546397413) q[3];
rz(2.3970018472646686) q[3];
ry(-3.116477851868299) q[4];
rz(2.1009771297670428) q[4];
ry(3.1360284633376647) q[5];
rz(-1.7840144936939633) q[5];
ry(-2.9399488088045502) q[6];
rz(-0.19515996710772662) q[6];
ry(0.5499174848512193) q[7];
rz(-1.5931215610739498) q[7];
ry(0.00072938658256394) q[8];
rz(-0.3937742804374381) q[8];
ry(2.9453718266967943) q[9];
rz(0.35337892796859127) q[9];
ry(-2.446412122048963) q[10];
rz(0.8054374444213594) q[10];
ry(2.025560157366428) q[11];
rz(-0.7459450184426664) q[11];
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
ry(-1.478528076700475) q[0];
rz(-1.014708682872837) q[0];
ry(-1.3606908364657262) q[1];
rz(-2.6435368323017743) q[1];
ry(1.5055449696643104) q[2];
rz(-2.0482306953383116) q[2];
ry(-2.517284412296333) q[3];
rz(-0.7337770446332637) q[3];
ry(-1.129729511020174) q[4];
rz(-0.5986565060457316) q[4];
ry(-0.14155090960269678) q[5];
rz(2.5641017662723784) q[5];
ry(-2.9136600222378823) q[6];
rz(3.1350793440544305) q[6];
ry(-1.5649987652761963) q[7];
rz(1.9683870047496423) q[7];
ry(-1.5675045088485424) q[8];
rz(0.00025052945246719185) q[8];
ry(-3.131941231272679) q[9];
rz(-1.1616012571921304) q[9];
ry(-0.03409050735458925) q[10];
rz(-1.296608697268149) q[10];
ry(0.2859096884685416) q[11];
rz(-1.8521924986298872) q[11];
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
ry(-2.400318588773964) q[0];
rz(0.7617283748077163) q[0];
ry(1.9709625975064906) q[1];
rz(-1.6338821848075904) q[1];
ry(1.2565487611264246) q[2];
rz(-0.27304349219247026) q[2];
ry(2.574084858733947) q[3];
rz(0.00232839955162234) q[3];
ry(-2.8523030304768557) q[4];
rz(-0.024462853534557606) q[4];
ry(2.631306000771264) q[5];
rz(-2.797665006507997) q[5];
ry(-1.7568221479122397) q[6];
rz(-0.00930440003525046) q[6];
ry(-0.005385010901958718) q[7];
rz(1.201083213126633) q[7];
ry(1.5133940165431268) q[8];
rz(0.849498269503268) q[8];
ry(-0.015912246246498188) q[9];
rz(-0.9356271498710309) q[9];
ry(0.7545519882698581) q[10];
rz(-2.4285592736385735) q[10];
ry(-2.2325606762679913) q[11];
rz(1.434007798804706) q[11];
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
ry(1.2235395429696725) q[0];
rz(-0.35646702932266544) q[0];
ry(-2.86622039743563) q[1];
rz(-0.20131946836781986) q[1];
ry(-0.0022042250162934303) q[2];
rz(-2.2722250174030765) q[2];
ry(1.0896302488785166) q[3];
rz(-3.1376315182743384) q[3];
ry(-2.8408292912373527) q[4];
rz(-2.8893911860134205) q[4];
ry(3.138274817109024) q[5];
rz(-2.590120040397287) q[5];
ry(-2.977896259535235) q[6];
rz(2.8285301326805827) q[6];
ry(1.8100109682032137) q[7];
rz(1.481324671777917) q[7];
ry(0.0006971957870502976) q[8];
rz(0.3726985039911775) q[8];
ry(0.018887361656029) q[9];
rz(-1.7552222025660695) q[9];
ry(-3.1230858224311437) q[10];
rz(-0.716186327620438) q[10];
ry(-1.0411541648877964) q[11];
rz(2.0703523790627694) q[11];
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
ry(-2.250195800518558) q[0];
rz(1.1594750254121937) q[0];
ry(-2.337520371828554) q[1];
rz(1.1993281462895364) q[1];
ry(-0.48502274463635064) q[2];
rz(-0.9892935152362005) q[2];
ry(0.5091262467599498) q[3];
rz(2.628669185534273) q[3];
ry(-3.1240635801597927) q[4];
rz(-2.8297093152408452) q[4];
ry(-2.99872246701226) q[5];
rz(2.680553906594485) q[5];
ry(3.131107418426369) q[6];
rz(0.6761209407438021) q[6];
ry(-2.416290698277484) q[7];
rz(2.6953523138839026) q[7];
ry(-3.1405620874411504) q[8];
rz(-2.016622436936755) q[8];
ry(-3.127767233055955) q[9];
rz(1.4601116031093984) q[9];
ry(1.7622720313230866) q[10];
rz(-0.18517135915400654) q[10];
ry(2.827554520335324) q[11];
rz(0.17317513545174548) q[11];
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
ry(0.43530539681291197) q[0];
rz(-0.6941907948320779) q[0];
ry(2.0510938414095055) q[1];
rz(-2.81233826793509) q[1];
ry(0.436608773072402) q[2];
rz(-0.27013949216679134) q[2];
ry(-1.9680344071958529) q[3];
rz(2.8201111223158772) q[3];
ry(-3.0812190337661645) q[4];
rz(-3.1119788287815258) q[4];
ry(-3.14029290172353) q[5];
rz(-1.673019934984914) q[5];
ry(0.0013754231644478308) q[6];
rz(2.156455338882248) q[6];
ry(0.017748796876300155) q[7];
rz(-2.973426668712188) q[7];
ry(-3.135210580730404) q[8];
rz(1.9830904301217123) q[8];
ry(2.489190349038019) q[9];
rz(-2.2300731837540106) q[9];
ry(-1.5679979570468154) q[10];
rz(-3.102164332551863) q[10];
ry(-0.8378102702196736) q[11];
rz(2.203030313014531) q[11];
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
ry(-1.615654715252119) q[0];
rz(-0.1706516364861968) q[0];
ry(0.13291603551303588) q[1];
rz(-1.6439842081377585) q[1];
ry(1.8326907937727346) q[2];
rz(-2.124731082340369) q[2];
ry(-0.4932441325845085) q[3];
rz(-2.1541765382083473) q[3];
ry(-1.7446172905293622) q[4];
rz(0.9292571740282238) q[4];
ry(0.028752211125200766) q[5];
rz(1.015049113679807) q[5];
ry(-1.230765818010152) q[6];
rz(1.4219978384580272) q[6];
ry(1.2451044638137008) q[7];
rz(1.8569696560826316) q[7];
ry(1.7412580810407619) q[8];
rz(-0.13598202165477702) q[8];
ry(0.08548479053422679) q[9];
rz(1.1396434138234968) q[9];
ry(-0.28447790583601146) q[10];
rz(0.8219447388232927) q[10];
ry(0.03313688127970149) q[11];
rz(-1.796501199984001) q[11];
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
ry(-0.9343903883075013) q[0];
rz(-2.6159006924919987) q[0];
ry(1.4987758791359618) q[1];
rz(-1.5894348982055675) q[1];
ry(-2.939418615690338) q[2];
rz(-1.3714012457462825) q[2];
ry(1.2286357139772408) q[3];
rz(-1.9101482005911947) q[3];
ry(0.2853673771560219) q[4];
rz(2.2324705020384723) q[4];
ry(0.17066195818067606) q[5];
rz(0.6871481928028311) q[5];
ry(-3.1396141091828667) q[6];
rz(1.4162847690493905) q[6];
ry(3.139124759202721) q[7];
rz(1.6192837524059327) q[7];
ry(-3.1404556837121445) q[8];
rz(-1.7629185264808473) q[8];
ry(-0.0003567301659678714) q[9];
rz(0.8823057781049721) q[9];
ry(-0.0016104456861771865) q[10];
rz(2.3060639872461355) q[10];
ry(-2.3642118284252946) q[11];
rz(-1.4868441490266067) q[11];
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
ry(-1.5939945792943726) q[0];
rz(-2.286330857793446) q[0];
ry(1.4728603302361374) q[1];
rz(1.8357533433783706) q[1];
ry(-3.087833379817939) q[2];
rz(-1.4270826280134337) q[2];
ry(-3.135200873282223) q[3];
rz(-1.8872310809616932) q[3];
ry(-3.1323462061788065) q[4];
rz(-2.588081970198948) q[4];
ry(-3.135765693018246) q[5];
rz(1.3555000873556189) q[5];
ry(1.377369025664441) q[6];
rz(1.96149441971464) q[6];
ry(2.6264457139177932) q[7];
rz(1.3151481932357165) q[7];
ry(2.078936519865788) q[8];
rz(-2.5035661763545276) q[8];
ry(1.6602539708111015) q[9];
rz(-3.039917718209867) q[9];
ry(-1.5132660849818114) q[10];
rz(2.807215728652122) q[10];
ry(-0.16529299276366055) q[11];
rz(3.135748850392085) q[11];
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
ry(-1.588841999814181) q[0];
rz(-0.16039747848112107) q[0];
ry(-1.9270135821558112) q[1];
rz(2.1210833723616833) q[1];
ry(1.5650346734721339) q[2];
rz(1.0997583246534264) q[2];
ry(1.897833757325285) q[3];
rz(-0.8728170360644779) q[3];
ry(0.9688064829518915) q[4];
rz(2.814741016408262) q[4];
ry(0.006199546151519187) q[5];
rz(-1.7022043597896173) q[5];
ry(-2.022672103923639) q[6];
rz(1.9108034245899537) q[6];
ry(-1.5637075318160378) q[7];
rz(1.5200612847676478) q[7];
ry(3.133108630006824) q[8];
rz(-0.9170303375413082) q[8];
ry(2.6149626176879663) q[9];
rz(1.0400488366931138) q[9];
ry(-3.136718956568054) q[10];
rz(-1.9436152420096207) q[10];
ry(-1.6214672551490201) q[11];
rz(0.20202666160919683) q[11];
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
ry(-1.469798035341163) q[0];
rz(0.06922083543532853) q[0];
ry(-1.5774309661622183) q[1];
rz(1.5224624365048498) q[1];
ry(1.5358639884536214) q[2];
rz(1.517093578578866) q[2];
ry(-3.1244279280297906) q[3];
rz(-1.0067217902795544) q[3];
ry(3.1025589397479094) q[4];
rz(-2.6999594939516207) q[4];
ry(3.1393075572829807) q[5];
rz(2.0283474912795185) q[5];
ry(3.1414117905992214) q[6];
rz(-1.5749945556359268) q[6];
ry(-0.0005842473716777263) q[7];
rz(0.06867102868949998) q[7];
ry(-3.1312477940044556) q[8];
rz(-0.16189208469843305) q[8];
ry(0.21951793759882943) q[9];
rz(0.9647383018313094) q[9];
ry(-0.1870722332380632) q[10];
rz(1.6045522800386338) q[10];
ry(1.6945853382398512) q[11];
rz(-3.1027708114463888) q[11];
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
ry(-3.0678052163549614) q[0];
rz(-1.6911704793653684) q[0];
ry(-3.12847696127693) q[1];
rz(1.541690709026068) q[1];
ry(1.5790540484255287) q[2];
rz(-0.01294505172701399) q[2];
ry(-0.020284667279748092) q[3];
rz(1.1260993759492393) q[3];
ry(-2.2534213781926073) q[4];
rz(-1.1970443636897288) q[4];
ry(3.1401054102350483) q[5];
rz(-1.6531815406272865) q[5];
ry(2.1191568867711608) q[6];
rz(-2.921721897508694) q[6];
ry(2.9779227859268187) q[7];
rz(-3.1249767850383865) q[7];
ry(1.4698505737313816) q[8];
rz(1.5764780007159906) q[8];
ry(-3.127740019835515) q[9];
rz(0.40005272621190074) q[9];
ry(-1.5698199654521168) q[10];
rz(1.5711299398268501) q[10];
ry(-1.6477779276971471) q[11];
rz(-1.9135962460601728) q[11];
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
ry(-1.4585010369639861) q[0];
rz(1.9356813636455286) q[0];
ry(2.875901625068188) q[1];
rz(0.9832470821907878) q[1];
ry(1.0878139102722493) q[2];
rz(-1.2245146861133824) q[2];
ry(1.5663042145289008) q[3];
rz(2.555617279133333) q[3];
ry(1.5764448782040699) q[4];
rz(-1.6816792633198698) q[4];
ry(1.569287274400356) q[5];
rz(-2.125907121995499) q[5];
ry(-1.5691439814257864) q[6];
rz(-0.0675527956991567) q[6];
ry(1.5700204258766464) q[7];
rz(1.0383988275041114) q[7];
ry(1.5701385142214466) q[8];
rz(-3.1042512298514557) q[8];
ry(-1.5708524123991452) q[9];
rz(2.6088378651865747) q[9];
ry(1.5708261128027905) q[10];
rz(3.1310381253484967) q[10];
ry(-3.1342539436663976) q[11];
rz(2.256892709899925) q[11];