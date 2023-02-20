OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-3.0881593738879616) q[0];
rz(3.0719740420959143) q[0];
ry(-1.1929811954628606) q[1];
rz(0.20130177300171956) q[1];
ry(-0.021162394236597278) q[2];
rz(-2.1265061438005404) q[2];
ry(-0.5534241742578591) q[3];
rz(-1.103725872120333) q[3];
ry(-0.15780898557684342) q[4];
rz(2.230955802333545) q[4];
ry(1.0561417415076664) q[5];
rz(0.010898034525467002) q[5];
ry(0.2644007737994515) q[6];
rz(1.8030493148428839) q[6];
ry(-2.2415913861868306) q[7];
rz(3.01413982111899) q[7];
ry(1.5769945479580876) q[8];
rz(1.3422974365701268) q[8];
ry(2.985757994840219) q[9];
rz(-2.493741508651268) q[9];
ry(-2.514272654266858) q[10];
rz(0.7349463811095029) q[10];
ry(1.3835768723110444) q[11];
rz(-2.5095960941270543) q[11];
ry(0.06591561364641983) q[12];
rz(2.999197194917522) q[12];
ry(-0.9128345040310917) q[13];
rz(-0.044876055812135725) q[13];
ry(-2.8928195265095087) q[14];
rz(2.724442846975144) q[14];
ry(0.12605013428644524) q[15];
rz(-0.048514151905189706) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.1390969513553615) q[0];
rz(1.3611767485676962) q[0];
ry(0.34077987807572274) q[1];
rz(-1.2924947974864336) q[1];
ry(2.510990760033974) q[2];
rz(-2.7277328967582712) q[2];
ry(-0.7043702347654676) q[3];
rz(2.1984455812249237) q[3];
ry(-0.035737153714378245) q[4];
rz(-3.0764597862036296) q[4];
ry(1.251004200743454) q[5];
rz(1.2952161269959397) q[5];
ry(-3.0409067928038023) q[6];
rz(0.03481496077908345) q[6];
ry(3.0557911751809104) q[7];
rz(-1.48954391640237) q[7];
ry(-2.9631086029261953) q[8];
rz(-2.395397642959182) q[8];
ry(3.1063922247700706) q[9];
rz(0.901018892478687) q[9];
ry(3.064168213316608) q[10];
rz(0.524302398917353) q[10];
ry(2.766784662291097) q[11];
rz(-0.42304683233808404) q[11];
ry(-0.1929452935726088) q[12];
rz(2.8129317542313186) q[12];
ry(2.13393440321806) q[13];
rz(-0.4411955906852656) q[13];
ry(1.1592649503973256) q[14];
rz(-0.6827911989192639) q[14];
ry(2.767716221096249) q[15];
rz(1.697320131828893) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.2048720352342843) q[0];
rz(1.2510285964575738) q[0];
ry(3.118084029352709) q[1];
rz(1.1699129290525354) q[1];
ry(0.016038617642626818) q[2];
rz(1.4687136723856193) q[2];
ry(2.6828917193771424) q[3];
rz(2.492664649207821) q[3];
ry(-0.06979323100694312) q[4];
rz(-0.9943723563077622) q[4];
ry(-1.2008599432266553) q[5];
rz(-1.4737205812416596) q[5];
ry(0.41509142992537534) q[6];
rz(1.484493735799474) q[6];
ry(0.2532606817481051) q[7];
rz(-0.5045942495015918) q[7];
ry(-0.72231367494153) q[8];
rz(2.161724494935969) q[8];
ry(2.9922692897457766) q[9];
rz(-0.22483155990547932) q[9];
ry(-2.4673891399651247) q[10];
rz(-2.8560772834814854) q[10];
ry(-0.30800654675796757) q[11];
rz(2.2893010418528204) q[11];
ry(0.20091962218667359) q[12];
rz(-0.49212151064976783) q[12];
ry(-0.08646351914214001) q[13];
rz(-0.5874428155796684) q[13];
ry(1.6521952377231197) q[14];
rz(-0.5996774502822723) q[14];
ry(-2.070974441190403) q[15];
rz(0.15211271012611596) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.971423412987573) q[0];
rz(2.8204402384138283) q[0];
ry(3.0269951714923375) q[1];
rz(0.8590155721772919) q[1];
ry(0.6112315162623184) q[2];
rz(0.714147316174409) q[2];
ry(-1.1627837462326367) q[3];
rz(-2.7025766444923787) q[3];
ry(3.0308172555391195) q[4];
rz(-1.9869633760371068) q[4];
ry(2.4536844896637837) q[5];
rz(1.232047838332127) q[5];
ry(-3.049189987020185) q[6];
rz(-0.8914534809766637) q[6];
ry(0.04218014403234491) q[7];
rz(-2.5420953070960914) q[7];
ry(-1.884174243880895) q[8];
rz(3.034314654037522) q[8];
ry(2.8183271093536835) q[9];
rz(-1.7300054667235791) q[9];
ry(-1.3900091089406252) q[10];
rz(-0.3089402476946606) q[10];
ry(-0.09708802629408195) q[11];
rz(0.617209252867836) q[11];
ry(-0.13998102692719347) q[12];
rz(-3.11260962899518) q[12];
ry(-0.18250575836163738) q[13];
rz(1.4909255431124022) q[13];
ry(0.8428151988119313) q[14];
rz(0.30534658280220217) q[14];
ry(2.100176420309399) q[15];
rz(-0.814766918519604) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.605755229739108) q[0];
rz(1.190542011860523) q[0];
ry(1.5939935438618225) q[1];
rz(-3.0846809252758027) q[1];
ry(3.1339325972280374) q[2];
rz(-3.0977202772426566) q[2];
ry(0.5875126897855645) q[3];
rz(1.63393198626836) q[3];
ry(-1.0244534767950733) q[4];
rz(1.007935574272099) q[4];
ry(-2.0719720936174664) q[5];
rz(-1.868116689194701) q[5];
ry(2.47115721091933) q[6];
rz(1.3089877886269754) q[6];
ry(-2.284001488529234) q[7];
rz(2.3882223046367503) q[7];
ry(0.6006228258594115) q[8];
rz(0.20064401627466213) q[8];
ry(-3.025183222352354) q[9];
rz(3.0302212986588763) q[9];
ry(0.5286220955652841) q[10];
rz(-1.420207215520084) q[10];
ry(3.0347760160976542) q[11];
rz(1.864369862314408) q[11];
ry(-2.9233436688403773) q[12];
rz(-1.7644911416412699) q[12];
ry(-3.068852524894775) q[13];
rz(-2.614897625395712) q[13];
ry(2.6850948960196956) q[14];
rz(-2.8726954361261767) q[14];
ry(-0.8054023185883386) q[15];
rz(-2.6609117668648485) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.4064615633139852) q[0];
rz(-1.6082048820684518) q[0];
ry(-1.1247165200581755) q[1];
rz(-2.911572172227793) q[1];
ry(2.600550483503431) q[2];
rz(1.5025508936283902) q[2];
ry(-2.1346847878923) q[3];
rz(2.8319332985394867) q[3];
ry(-0.09022294137159825) q[4];
rz(-1.4766277508882322) q[4];
ry(-0.3493074346541074) q[5];
rz(-1.8434979925473192) q[5];
ry(-1.064725619069761) q[6];
rz(3.0339973139218093) q[6];
ry(-0.621447881557712) q[7];
rz(3.046838668742363) q[7];
ry(1.4487304992630183) q[8];
rz(0.70531171792579) q[8];
ry(-1.189868772188169) q[9];
rz(0.5664872355414287) q[9];
ry(-2.378972721294164) q[10];
rz(2.530638001308877) q[10];
ry(0.09578142536444112) q[11];
rz(-1.8543636796608975) q[11];
ry(0.19727854905561948) q[12];
rz(2.275316281755763) q[12];
ry(0.43162084570378045) q[13];
rz(-0.6041836127892823) q[13];
ry(0.9220468551053251) q[14];
rz(-0.3281229687235731) q[14];
ry(2.7351292141364296) q[15];
rz(-1.2956953773443713) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.464809838451508) q[0];
rz(2.2440140441876) q[0];
ry(3.0778075924732144) q[1];
rz(-3.101037090254911) q[1];
ry(-0.009084437864477003) q[2];
rz(2.575065508620246) q[2];
ry(0.551098221733008) q[3];
rz(-2.2068462516244187) q[3];
ry(-1.4008621408651454) q[4];
rz(-1.5944797692522927) q[4];
ry(-0.02725523125377283) q[5];
rz(1.6820793897137811) q[5];
ry(0.013957416620981088) q[6];
rz(0.8246694778614057) q[6];
ry(1.504425824865764) q[7];
rz(0.07631456065014462) q[7];
ry(2.34952686004595) q[8];
rz(-0.09999685171528672) q[8];
ry(2.0999514046020433) q[9];
rz(0.9041942615742584) q[9];
ry(1.4757051598850524) q[10];
rz(0.648458153235203) q[10];
ry(-0.9914866827143671) q[11];
rz(-2.4433848417287667) q[11];
ry(3.0323917070077457) q[12];
rz(1.1122026036474253) q[12];
ry(0.3181570091067565) q[13];
rz(0.2491478586588469) q[13];
ry(0.158171687239288) q[14];
rz(-1.9435867582970248) q[14];
ry(0.16762400856395326) q[15];
rz(1.1432186836316693) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.0106104654094636) q[0];
rz(-1.895335664693917) q[0];
ry(-1.9793726483097904) q[1];
rz(2.9317850587092056) q[1];
ry(1.9125087104456282) q[2];
rz(-0.1621196352049683) q[2];
ry(0.6369401668588752) q[3];
rz(3.113261379446607) q[3];
ry(2.556422326425292) q[4];
rz(1.2671922975783945) q[4];
ry(-0.18899422250402242) q[5];
rz(-2.6710063567180007) q[5];
ry(-1.6271425281191618) q[6];
rz(0.024706030234012744) q[6];
ry(1.1946275640599728) q[7];
rz(-2.500310000575731) q[7];
ry(3.053958405500817) q[8];
rz(-0.21110182778839776) q[8];
ry(-0.05364500386505) q[9];
rz(2.589227642104104) q[9];
ry(-0.11860971715588953) q[10];
rz(-1.5502869179223229) q[10];
ry(-0.0935211631776544) q[11];
rz(2.7318480787228503) q[11];
ry(0.012164078289889524) q[12];
rz(-0.9465162753805626) q[12];
ry(-2.6606831431119855) q[13];
rz(-0.7307972313622564) q[13];
ry(-1.9755568005954454) q[14];
rz(2.861237527239264) q[14];
ry(2.193458454323255) q[15];
rz(1.7653403926062161) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.9888373790228693) q[0];
rz(2.1332980349875577) q[0];
ry(-0.022427701274007283) q[1];
rz(-2.854275399107726) q[1];
ry(-3.026434087892269) q[2];
rz(-2.714899047833248) q[2];
ry(0.3290196711371687) q[3];
rz(2.940843641047068) q[3];
ry(0.29063931036818413) q[4];
rz(2.666718788124345) q[4];
ry(1.1450480005352957) q[5];
rz(0.746810800966747) q[5];
ry(0.04364587565087166) q[6];
rz(-2.1393768241378606) q[6];
ry(2.813660523348575) q[7];
rz(-1.561632184531752) q[7];
ry(-1.39358762540494) q[8];
rz(-0.8662473006745912) q[8];
ry(1.167543770513017) q[9];
rz(-2.0538886525387117) q[9];
ry(-1.360599847326098) q[10];
rz(0.5313417960317235) q[10];
ry(-0.16280664682898038) q[11];
rz(3.081074337494588) q[11];
ry(3.0802787021493256) q[12];
rz(0.8805262307820279) q[12];
ry(-2.5054998512621327) q[13];
rz(-2.8454052899977946) q[13];
ry(0.23689717055224513) q[14];
rz(0.8408296188255616) q[14];
ry(-0.10201907625301754) q[15];
rz(2.316516931708923) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.552683382360775) q[0];
rz(-0.9051147140846133) q[0];
ry(0.7068164441182843) q[1];
rz(1.616320057292082) q[1];
ry(1.928488053948561) q[2];
rz(2.92756749974955) q[2];
ry(-0.45432036178401003) q[3];
rz(-2.015952385920855) q[3];
ry(2.013600143500967) q[4];
rz(-0.9337197336437093) q[4];
ry(-0.8409690199349855) q[5];
rz(-2.0323501498288365) q[5];
ry(-0.05277915133046947) q[6];
rz(-2.097730073284202) q[6];
ry(2.4723350890086855) q[7];
rz(-2.4933001874216822) q[7];
ry(2.2849423214158477) q[8];
rz(-1.8127142330541455) q[8];
ry(-0.39837802191608507) q[9];
rz(2.8634226273744883) q[9];
ry(-1.5893599279066775) q[10];
rz(-0.11488039105352055) q[10];
ry(2.550832296137483) q[11];
rz(1.0774572803219262) q[11];
ry(-0.03218813787384889) q[12];
rz(-1.8059870710386126) q[12];
ry(-2.8492235652420677) q[13];
rz(2.8202059866105227) q[13];
ry(1.7789379355656774) q[14];
rz(1.47888324350256) q[14];
ry(-1.8899588617964547) q[15];
rz(-2.791786029727649) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.8201708050647487) q[0];
rz(-0.6083146180804757) q[0];
ry(-3.024663114192805e-05) q[1];
rz(-1.5837699354218218) q[1];
ry(0.007352376196767985) q[2];
rz(1.5481909374422116) q[2];
ry(-0.28391365658540657) q[3];
rz(-1.59903462490585) q[3];
ry(0.8576365878462502) q[4];
rz(2.6775220109012605) q[4];
ry(0.7485885050545233) q[5];
rz(0.3141845098589746) q[5];
ry(-2.2418362796538016) q[6];
rz(2.6032946792083176) q[6];
ry(-0.09730661694353594) q[7];
rz(2.609791812345821) q[7];
ry(-1.1276740700570755) q[8];
rz(-0.002579338317329135) q[8];
ry(-2.712134936588075) q[9];
rz(-1.3192439556121194) q[9];
ry(0.5383809665646622) q[10];
rz(1.2802874893007847) q[10];
ry(-2.663862048376274) q[11];
rz(-2.3020215270350284) q[11];
ry(-0.36288975457740114) q[12];
rz(-2.7963812393719656) q[12];
ry(-0.0233460213567982) q[13];
rz(2.70111178393651) q[13];
ry(2.976110313441438) q[14];
rz(-0.8649266179085808) q[14];
ry(-3.0229494679025435) q[15];
rz(-2.386299758761792) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.1627033039984447) q[0];
rz(-0.21213107082308366) q[0];
ry(-0.3846573258533517) q[1];
rz(-0.6567790853260789) q[1];
ry(0.9654849818217787) q[2];
rz(1.974987327295257) q[2];
ry(0.02501438810922973) q[3];
rz(-0.8501667569734757) q[3];
ry(-3.003510634688478) q[4];
rz(1.3476893431194856) q[4];
ry(2.2590232586504984) q[5];
rz(-2.5417098614847795) q[5];
ry(-3.109698162126667) q[6];
rz(-1.5944960394221175) q[6];
ry(-3.116159092205891) q[7];
rz(-2.4266736488201057) q[7];
ry(-1.395282968599991) q[8];
rz(3.0989447269867236) q[8];
ry(-0.10148068990258442) q[9];
rz(1.0411061886698776) q[9];
ry(2.9715062747853485) q[10];
rz(1.6498136335779117) q[10];
ry(-0.9359112624907292) q[11];
rz(0.14529146591300635) q[11];
ry(0.04767155683413815) q[12];
rz(-1.5080141124687803) q[12];
ry(-3.0968015976791072) q[13];
rz(1.193781968819653) q[13];
ry(-0.1641057796211838) q[14];
rz(-2.912955668077187) q[14];
ry(-2.3357919294928964) q[15];
rz(-0.7493295909069664) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.6867540332042941) q[0];
rz(-2.4603116832217395) q[0];
ry(0.03934479968957017) q[1];
rz(-1.983445468762306) q[1];
ry(-2.9167319014464814) q[2];
rz(0.8366991612928563) q[2];
ry(-1.1936250827229644) q[3];
rz(-0.5616812375162193) q[3];
ry(2.491661639545037) q[4];
rz(-0.5465133680794124) q[4];
ry(-2.1266264230828638) q[5];
rz(-0.20406698998660658) q[5];
ry(1.3171103679331004) q[6];
rz(-1.24055439238915) q[6];
ry(-0.12323236454834063) q[7];
rz(-3.034078322310082) q[7];
ry(1.4853478443321024) q[8];
rz(2.061405534004715) q[8];
ry(0.6916548239701319) q[9];
rz(1.6289886386638397) q[9];
ry(-1.2348668432719154) q[10];
rz(-2.772816359972939) q[10];
ry(-0.2578379991727093) q[11];
rz(-2.3740933348855338) q[11];
ry(0.5508987085771073) q[12];
rz(2.5459259547568154) q[12];
ry(-2.851356740336239) q[13];
rz(-1.6247974704281543) q[13];
ry(-0.07564670788759997) q[14];
rz(1.5761085522450582) q[14];
ry(-0.6375638018077696) q[15];
rz(-2.591655456789808) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.5621822809165193) q[0];
rz(-0.10602333367824228) q[0];
ry(1.5443132035040403) q[1];
rz(-0.7867628843800825) q[1];
ry(2.864241101111193) q[2];
rz(-1.9150262168314323) q[2];
ry(-0.08488447231458254) q[3];
rz(-2.867320310424749) q[3];
ry(-0.7031071709417809) q[4];
rz(-3.0580661839234407) q[4];
ry(1.3387507064438218) q[5];
rz(0.5936743788947755) q[5];
ry(3.09634795373949) q[6];
rz(-1.0347658331592688) q[6];
ry(-0.00059475869745265) q[7];
rz(1.980604845006036) q[7];
ry(-0.7339292274642384) q[8];
rz(1.9197631620908306) q[8];
ry(-3.1336228860149045) q[9];
rz(-2.2338758205258955) q[9];
ry(0.4209297111231482) q[10];
rz(0.2687612865840936) q[10];
ry(3.1196591716430526) q[11];
rz(-0.05881290038575397) q[11];
ry(-0.030176602442622883) q[12];
rz(0.6608114939595893) q[12];
ry(0.002072015990361017) q[13];
rz(-3.0549256103803297) q[13];
ry(-1.517441470964986) q[14];
rz(0.0012779636915576376) q[14];
ry(-1.9486492660754253) q[15];
rz(2.159048754000321) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.02549250603433338) q[0];
rz(3.091562278932447) q[0];
ry(-3.1357154555987354) q[1];
rz(2.209210785908371) q[1];
ry(-0.1890087080122429) q[2];
rz(-2.800414758900426) q[2];
ry(2.14316301833058) q[3];
rz(-0.5889947852659336) q[3];
ry(-1.147624421998847) q[4];
rz(1.4101341138581165) q[4];
ry(0.35708442914176775) q[5];
rz(0.0361933739371295) q[5];
ry(1.4374126473236615) q[6];
rz(-1.3372674985728052) q[6];
ry(1.006594618481095) q[7];
rz(1.1852696899371078) q[7];
ry(-1.0503730924094006) q[8];
rz(2.2716395490335675) q[8];
ry(-2.9860837825133384) q[9];
rz(-1.2333902592667112) q[9];
ry(-0.4805371452796297) q[10];
rz(2.385651347573044) q[10];
ry(-0.04284949240183078) q[11];
rz(-1.5608298055537513) q[11];
ry(2.044600584418519) q[12];
rz(2.432105929814412) q[12];
ry(-2.224104106369139) q[13];
rz(-1.7343519903663651) q[13];
ry(-0.16394174702359446) q[14];
rz(-2.9268691954630555) q[14];
ry(-1.0267077718974917) q[15];
rz(1.8259620164120405) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.4675459029442104) q[0];
rz(-3.109886698443454) q[0];
ry(1.700598504853196) q[1];
rz(-1.277897041084388) q[1];
ry(2.4190151990066098) q[2];
rz(0.5823267530895846) q[2];
ry(-0.27973669273308294) q[3];
rz(-2.2374116736357097) q[3];
ry(-2.1881861348513985) q[4];
rz(0.3856935605037222) q[4];
ry(-0.6138447548840675) q[5];
rz(-1.352449759711197) q[5];
ry(-0.0058090403524538705) q[6];
rz(-1.6537159474901524) q[6];
ry(-3.078679192301288) q[7];
rz(2.603296937689523) q[7];
ry(2.4935993735672715) q[8];
rz(2.6351732263429106) q[8];
ry(-0.05953836941493229) q[9];
rz(-0.01305299797296744) q[9];
ry(-2.0132107265338455) q[10];
rz(0.1963728325637728) q[10];
ry(-0.2913466200447523) q[11];
rz(2.6911719012313617) q[11];
ry(0.04339870503251219) q[12];
rz(-0.8018792773384217) q[12];
ry(3.0702310258890844) q[13];
rz(-2.52265676068107) q[13];
ry(-1.5435940040775993) q[14];
rz(-3.129319831624803) q[14];
ry(-2.0088627965134838) q[15];
rz(0.11142629875648602) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.03528489235623925) q[0];
rz(2.9126690024629807) q[0];
ry(3.127711236425205) q[1];
rz(1.6145306138028213) q[1];
ry(-0.1258686526164876) q[2];
rz(0.36846464640889653) q[2];
ry(-2.3810095937650826) q[3];
rz(-1.459957550873391) q[3];
ry(-1.7753087165324217) q[4];
rz(0.7945299737465669) q[4];
ry(-1.5153323867730932) q[5];
rz(2.7215722394253814) q[5];
ry(0.7654357320159532) q[6];
rz(1.0322486757758744) q[6];
ry(-0.20515541101360887) q[7];
rz(0.08560974880869843) q[7];
ry(-0.9705975964567415) q[8];
rz(-0.2731955467274414) q[8];
ry(-0.5360656709611336) q[9];
rz(0.5456736588098743) q[9];
ry(-1.7202169814898483) q[10];
rz(-1.0008254764324436) q[10];
ry(-2.8156300140615174) q[11];
rz(-2.2746035374053415) q[11];
ry(-0.7159675384459554) q[12];
rz(-0.931596833950795) q[12];
ry(1.7039684947910176) q[13];
rz(-2.7985382782409407) q[13];
ry(0.558577161973771) q[14];
rz(-2.3586583915621784) q[14];
ry(-3.029359379273696) q[15];
rz(0.6572445604308232) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.3428461313880611) q[0];
rz(1.3190667503518907) q[0];
ry(-2.304171478163584) q[1];
rz(-0.6605896730005426) q[1];
ry(0.6424929108758781) q[2];
rz(0.9389057391404972) q[2];
ry(-0.4767032118097175) q[3];
rz(-2.7180345925802656) q[3];
ry(2.9806633260805286) q[4];
rz(-0.32260436003442755) q[4];
ry(-0.7547269296426631) q[5];
rz(-2.1112585553092478) q[5];
ry(-3.130318263924101) q[6];
rz(-1.3028198880042952) q[6];
ry(-0.04539199904749375) q[7];
rz(2.431690345729358) q[7];
ry(-0.06452798515428704) q[8];
rz(0.6249721196805336) q[8];
ry(-0.018756456877421) q[9];
rz(-0.8906231086436049) q[9];
ry(2.2869533184660815) q[10];
rz(-1.9032992155428303) q[10];
ry(-2.764588042091948) q[11];
rz(1.0303496852365184) q[11];
ry(-2.8755827016699897) q[12];
rz(2.4736801920146005) q[12];
ry(0.07025400234083444) q[13];
rz(2.674709679788972) q[13];
ry(2.4566822140940467) q[14];
rz(2.754543997980297) q[14];
ry(-2.9480742424994255) q[15];
rz(0.33091066276710557) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.06698625948725301) q[0];
rz(3.09924095488315) q[0];
ry(-2.1183864457568937) q[1];
rz(1.5065698613641993) q[1];
ry(0.31962333301740653) q[2];
rz(-0.2592849070121418) q[2];
ry(-0.0560999868301062) q[3];
rz(1.5014435552995384) q[3];
ry(-1.8861166061776016) q[4];
rz(-2.684476863032277) q[4];
ry(0.8741889506017453) q[5];
rz(-1.2425057817207292) q[5];
ry(-1.5455090008208534) q[6];
rz(2.362213546722011) q[6];
ry(1.4740235735620404) q[7];
rz(-0.18262605134071414) q[7];
ry(-0.18848182623609144) q[8];
rz(0.9449845843612836) q[8];
ry(0.9113115688199191) q[9];
rz(0.005743498243957822) q[9];
ry(1.5918383267050764) q[10];
rz(3.1087683416666034) q[10];
ry(3.071458905498293) q[11];
rz(-2.0113084786679636) q[11];
ry(1.478914622203991) q[12];
rz(0.8751635216637004) q[12];
ry(2.836847808759219) q[13];
rz(0.33747973296527434) q[13];
ry(2.534937678522773) q[14];
rz(3.1096852735441702) q[14];
ry(2.6890008436793265) q[15];
rz(-2.252708345924675) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.1323773832240573) q[0];
rz(0.8177359032148619) q[0];
ry(1.5091859328415296) q[1];
rz(-0.1229027908627546) q[1];
ry(-1.3429744217121045) q[2];
rz(1.1286025599454312) q[2];
ry(-1.564322795384439) q[3];
rz(2.8835235797770538) q[3];
ry(-0.6246370167482871) q[4];
rz(-2.258089908122781) q[4];
ry(-0.23990031930134512) q[5];
rz(-2.7409414567885726) q[5];
ry(1.026819342342627) q[6];
rz(-0.7218894230744236) q[6];
ry(-2.7629787842128355) q[7];
rz(1.7569515833753133) q[7];
ry(-0.19136721436897636) q[8];
rz(-0.0282844329798122) q[8];
ry(-0.020246713830491838) q[9];
rz(2.842272794898366) q[9];
ry(1.605485332778679) q[10];
rz(-0.08245300421920129) q[10];
ry(-0.6554908943187927) q[11];
rz(-2.7947757985140154) q[11];
ry(-1.0905984186059405) q[12];
rz(2.1248232871540655) q[12];
ry(2.511573693103563) q[13];
rz(0.17853311698009636) q[13];
ry(-2.7940435551575558) q[14];
rz(2.875848537831364) q[14];
ry(-1.4846084418305736) q[15];
rz(1.8482954590558456) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.1203229596538797) q[0];
rz(-0.7005543463234415) q[0];
ry(0.3016289550339462) q[1];
rz(-1.8805458418357421) q[1];
ry(3.1353232894004797) q[2];
rz(-0.574395352326201) q[2];
ry(-0.06464233101205874) q[3];
rz(-1.3899485450397002) q[3];
ry(-1.148150693254782) q[4];
rz(-1.8978649639648797) q[4];
ry(3.125686808562526) q[5];
rz(-0.07750671888606628) q[5];
ry(-0.0005015506609014426) q[6];
rz(0.7311720155750879) q[6];
ry(-0.04657183923456392) q[7];
rz(0.3123265743779217) q[7];
ry(0.14852435998390678) q[8];
rz(-0.1624933078911761) q[8];
ry(-1.4766941743838042) q[9];
rz(-0.8423451578570562) q[9];
ry(-2.364711302567652) q[10];
rz(2.679435854856682) q[10];
ry(-1.5283290646616452) q[11];
rz(1.4451319703581422) q[11];
ry(-0.002954785566442584) q[12];
rz(2.668403455599591) q[12];
ry(2.9205974032374873) q[13];
rz(1.1315542821981515) q[13];
ry(-2.7601912226120193) q[14];
rz(2.1233384318109496) q[14];
ry(-0.7761122933130413) q[15];
rz(0.81052016193457) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.1350814907603666) q[0];
rz(-2.7099233986763402) q[0];
ry(-1.942722533292227) q[1];
rz(1.7481035866934127) q[1];
ry(-2.745225539573238) q[2];
rz(1.0082943416138215) q[2];
ry(-1.8411239082362336) q[3];
rz(-0.8964143939099054) q[3];
ry(0.4815419396531073) q[4];
rz(-2.8942348792206793) q[4];
ry(-0.1075476389056103) q[5];
rz(-3.1261255021243963) q[5];
ry(-2.1688152167637935) q[6];
rz(0.8065887209377358) q[6];
ry(-2.805241584282252) q[7];
rz(0.77345795042476) q[7];
ry(-0.32208937597979403) q[8];
rz(0.0061616348330599795) q[8];
ry(-3.087758142175656) q[9];
rz(1.8137534384898093) q[9];
ry(-0.0011070861896032014) q[10];
rz(-0.8278098794904577) q[10];
ry(0.011120700152337919) q[11];
rz(-2.8362662631186946) q[11];
ry(1.2830377791771523) q[12];
rz(-2.504256085081559) q[12];
ry(-0.6423989571232109) q[13];
rz(0.8731820405607936) q[13];
ry(-0.5415007991567491) q[14];
rz(-1.0198103227192652) q[14];
ry(-2.429997762705829) q[15];
rz(1.0959844739638473) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.8699393404688474) q[0];
rz(-2.912616720472342) q[0];
ry(2.8254837745317767) q[1];
rz(3.0734600776232814) q[1];
ry(0.22880009025978953) q[2];
rz(-2.21824318192695) q[2];
ry(3.0839864015848955) q[3];
rz(-2.10053589910949) q[3];
ry(-2.4788045665359446) q[4];
rz(2.110248083504805) q[4];
ry(0.3780153786404686) q[5];
rz(-1.805795533039939) q[5];
ry(-0.3660988817676604) q[6];
rz(-0.5527521972019409) q[6];
ry(2.7674033881860676) q[7];
rz(2.1506253369947923) q[7];
ry(2.836688046784648) q[8];
rz(-0.024091673647866685) q[8];
ry(2.2423566419785708) q[9];
rz(1.2826131489489925) q[9];
ry(2.00822369472482) q[10];
rz(-2.2677692929092426) q[10];
ry(2.150594688503196) q[11];
rz(-1.5297806750881335) q[11];
ry(0.03382379288157715) q[12];
rz(-0.7540635040210403) q[12];
ry(-3.1378972960718645) q[13];
rz(2.3813564882309826) q[13];
ry(0.8153505725272261) q[14];
rz(0.4132983548395201) q[14];
ry(-0.11598368214997201) q[15];
rz(1.6182703647977486) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.6491093943308568) q[0];
rz(-1.8241078697935231) q[0];
ry(0.9777088784812848) q[1];
rz(0.8673254898230253) q[1];
ry(-2.173000048069058) q[2];
rz(-0.6235457744012424) q[2];
ry(1.8105677837709466) q[3];
rz(-0.36809812849910306) q[3];
ry(2.539992330473661) q[4];
rz(-1.2393184204947303) q[4];
ry(-3.0962861323756816) q[5];
rz(-2.9341849130013067) q[5];
ry(-0.06491381093460702) q[6];
rz(2.9184113767502993) q[6];
ry(-0.5969430904936079) q[7];
rz(2.322861183938889) q[7];
ry(-2.7960835461839793) q[8];
rz(-1.5411321791713508) q[8];
ry(-0.05575490814656181) q[9];
rz(0.1272142908715241) q[9];
ry(3.131000766120283) q[10];
rz(-1.1184588117186287) q[10];
ry(-0.49318127562827563) q[11];
rz(-1.8970804714424192) q[11];
ry(2.7503458365911806) q[12];
rz(3.09129996836846) q[12];
ry(1.6826109861866714) q[13];
rz(0.5746491237303982) q[13];
ry(-1.165474154431147) q[14];
rz(-2.955278297678778) q[14];
ry(-0.020598009763900298) q[15];
rz(-2.2687473825347957) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.846005294067175) q[0];
rz(-0.3970129076977278) q[0];
ry(-3.1348439580986525) q[1];
rz(-2.125271706151172) q[1];
ry(-1.001680028571851) q[2];
rz(1.9579193126724261) q[2];
ry(1.4620822985226383) q[3];
rz(-2.85049284029581) q[3];
ry(0.8584273291406658) q[4];
rz(2.309797355414361) q[4];
ry(2.1690512577390964) q[5];
rz(0.7817232114121352) q[5];
ry(-0.1751331690741189) q[6];
rz(2.132949728744267) q[6];
ry(-0.1836750534995932) q[7];
rz(0.20457356995127007) q[7];
ry(0.19047801535603792) q[8];
rz(-2.5210237234395265) q[8];
ry(-1.9686498173605962) q[9];
rz(-2.9966459483394225) q[9];
ry(-3.046681137575678) q[10];
rz(1.0841019924841877) q[10];
ry(-0.8279316551440129) q[11];
rz(0.21972858384528318) q[11];
ry(-0.6526838525675603) q[12];
rz(2.045538958081739) q[12];
ry(-3.1376413730138557) q[13];
rz(-2.0780920714081255) q[13];
ry(1.9422633706115513) q[14];
rz(2.7217703914223104) q[14];
ry(-1.4277928027263878) q[15];
rz(-0.01931763275567988) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.7972138551624646) q[0];
rz(2.263857192870557) q[0];
ry(0.06956669972671126) q[1];
rz(0.15814821218863803) q[1];
ry(-0.14708638723070067) q[2];
rz(1.1004299573090905) q[2];
ry(0.010284668569405575) q[3];
rz(-1.883082205662047) q[3];
ry(-2.7361787797381067) q[4];
rz(1.876560612135794) q[4];
ry(-3.063710900338497) q[5];
rz(1.6454066807366672) q[5];
ry(-0.04420819038220536) q[6];
rz(-0.44058532006822654) q[6];
ry(-0.3910734613678697) q[7];
rz(-0.9110704008728713) q[7];
ry(-3.0063925402291747) q[8];
rz(-1.435047405192443) q[8];
ry(0.07332405638949537) q[9];
rz(2.6149319954161245) q[9];
ry(-3.114591571198869) q[10];
rz(-1.173360935280277) q[10];
ry(-0.13664222286840833) q[11];
rz(1.6760365530668526) q[11];
ry(-3.0236296495092114) q[12];
rz(-1.0893549991571332) q[12];
ry(2.413961700371919) q[13];
rz(-0.9420978280608315) q[13];
ry(2.910390575326934) q[14];
rz(0.49294687256918895) q[14];
ry(3.0991104753953724) q[15];
rz(0.7965549925429078) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.1402224798200913) q[0];
rz(1.944535390453356) q[0];
ry(2.3394709143000774) q[1];
rz(-1.224413051803873) q[1];
ry(2.2903037730531444) q[2];
rz(-0.4476041485116271) q[2];
ry(1.418696617469366) q[3];
rz(1.8719940074193202) q[3];
ry(0.35495231975547276) q[4];
rz(2.898868764177733) q[4];
ry(0.20671318545269912) q[5];
rz(-0.810239773870955) q[5];
ry(-1.2505853275128445) q[6];
rz(-1.775194567087377) q[6];
ry(-1.931689954255194) q[7];
rz(1.7800055555532897) q[7];
ry(1.6500818941594126) q[8];
rz(-2.62728403074768) q[8];
ry(2.7512650353651074) q[9];
rz(0.7369113029679646) q[9];
ry(-1.2934689716761207) q[10];
rz(1.4084642440852386) q[10];
ry(2.6071366134230654) q[11];
rz(2.9767720563331976) q[11];
ry(-0.07663082041268912) q[12];
rz(1.5482528079588767) q[12];
ry(3.1401744200578525) q[13];
rz(-1.664000881157163) q[13];
ry(-1.4842648225811113) q[14];
rz(-1.3709676328478806) q[14];
ry(-1.4355768082648965) q[15];
rz(-1.1076768902535716) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.830365224766634) q[0];
rz(-0.8558830972998521) q[0];
ry(3.007251296153013) q[1];
rz(1.2848519550285729) q[1];
ry(-3.137107907632073) q[2];
rz(-2.2153624478019154) q[2];
ry(0.24902052400658542) q[3];
rz(1.1761691109812995) q[3];
ry(-0.06876300137179807) q[4];
rz(2.2982469960624807) q[4];
ry(0.38554302120451817) q[5];
rz(-2.7474002152096477) q[5];
ry(0.059321383720570636) q[6];
rz(1.1211739043027897) q[6];
ry(-0.11342762998397958) q[7];
rz(-3.076431198193244) q[7];
ry(-1.5458549383847269) q[8];
rz(-3.1019742246892426) q[8];
ry(3.1407064171434254) q[9];
rz(2.7940021929231613) q[9];
ry(-0.1046804330098034) q[10];
rz(1.674166326735216) q[10];
ry(-3.0436497759864567) q[11];
rz(-1.6307536107669076) q[11];
ry(-0.10404867239673088) q[12];
rz(1.4944078303800876) q[12];
ry(-2.025703259307959) q[13];
rz(2.8519383953944053) q[13];
ry(-0.4844671635458919) q[14];
rz(-0.6448077570540525) q[14];
ry(-1.1336574959809982) q[15];
rz(2.4507578775660535) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.555983511352551) q[0];
rz(-2.2641680326628917) q[0];
ry(-0.4268366615694139) q[1];
rz(0.6502029498015449) q[1];
ry(-1.677999316757417) q[2];
rz(1.547955625777714) q[2];
ry(1.7005515513622784) q[3];
rz(2.643816678603934) q[3];
ry(-0.46946499096496064) q[4];
rz(2.3964095147929587) q[4];
ry(0.22135155409699872) q[5];
rz(2.5458438794371996) q[5];
ry(3.0709891165216883) q[6];
rz(0.29220676944580415) q[6];
ry(-0.03694623899326732) q[7];
rz(-1.2569979529697435) q[7];
ry(-1.5857469513484093) q[8];
rz(-0.05553052484569854) q[8];
ry(0.00736973542616643) q[9];
rz(0.9714803069193427) q[9];
ry(1.7893074594493994) q[10];
rz(0.5445835529571726) q[10];
ry(-1.5651052482097867) q[11];
rz(-0.8885851561216356) q[11];
ry(2.6876981145328465) q[12];
rz(1.5954155776716121) q[12];
ry(1.5685500175498683) q[13];
rz(3.1269842527587293) q[13];
ry(0.8171826764767554) q[14];
rz(3.0618785012538825) q[14];
ry(0.4205012113514188) q[15];
rz(1.2893097945917305) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.5855624456614553) q[0];
rz(-3.0128957107827365) q[0];
ry(1.4975719106215868) q[1];
rz(-1.5669381964781426) q[1];
ry(0.030726022016887327) q[2];
rz(0.17167661034847065) q[2];
ry(-3.1062436035165626) q[3];
rz(-0.0870365412585743) q[3];
ry(2.753526605021662) q[4];
rz(1.5986516443515226) q[4];
ry(0.45104364251232193) q[5];
rz(1.7251926547124474) q[5];
ry(0.02345050288609407) q[6];
rz(-1.4766031099823842) q[6];
ry(0.023837903928952464) q[7];
rz(2.5584971643626444) q[7];
ry(1.524555192702863) q[8];
rz(-1.661124617791353) q[8];
ry(-0.015760627365996704) q[9];
rz(0.09982136039616575) q[9];
ry(-0.05441638030852669) q[10];
rz(1.0049661755212966) q[10];
ry(-0.08525343080723366) q[11];
rz(0.9364336133370234) q[11];
ry(3.075815804919416) q[12];
rz(-3.0246189760874773) q[12];
ry(0.09819172957230204) q[13];
rz(1.5831478997085429) q[13];
ry(-1.570684965448022) q[14];
rz(1.568607076388658) q[14];
ry(2.018835366508731) q[15];
rz(-2.8441618551598884) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.011089435884364287) q[0];
rz(-2.502888426365757) q[0];
ry(1.5779791465803417) q[1];
rz(-2.3748303583053594) q[1];
ry(-1.5702871920119819) q[2];
rz(0.1713934856159576) q[2];
ry(-0.08852989427023886) q[3];
rz(2.9207524658796795) q[3];
ry(-0.7588701997369643) q[4];
rz(0.9450814314550783) q[4];
ry(-1.5588516926949845) q[5];
rz(-1.2393728780472655) q[5];
ry(-1.579428601070749) q[6];
rz(-1.1439721053163572) q[6];
ry(1.7517933709017597) q[7];
rz(-2.483470197146042) q[7];
ry(0.467931760710436) q[8];
rz(2.1550398913081223) q[8];
ry(-1.5754147400777097) q[9];
rz(1.9913034758632464) q[9];
ry(3.0918986426408015) q[10];
rz(-0.5425946634423404) q[10];
ry(3.101571757297556) q[11];
rz(1.306047459179224) q[11];
ry(1.5153960991231643) q[12];
rz(-0.7895216129088625) q[12];
ry(-1.5709238219792736) q[13];
rz(2.716704893954608) q[13];
ry(1.5699930124861403) q[14];
rz(1.659676850779389) q[14];
ry(-0.0036876264306933133) q[15];
rz(0.1997390341185149) q[15];