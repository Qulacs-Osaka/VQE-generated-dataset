OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.2639966428982303) q[0];
rz(-0.4013126303888557) q[0];
ry(2.518580304646344) q[1];
rz(-2.439196125081697) q[1];
ry(-2.3749236474096813) q[2];
rz(-2.07764654207272) q[2];
ry(-2.2240757486683265) q[3];
rz(2.2755879948833413) q[3];
ry(1.2133627495109518) q[4];
rz(2.3666744232435755) q[4];
ry(-2.7912549261768245) q[5];
rz(-1.2682956425466303) q[5];
ry(-2.076073530233505) q[6];
rz(0.14033099185844122) q[6];
ry(-3.042993400185002) q[7];
rz(2.0565600435901557) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.2071804768658714) q[0];
rz(0.47795856804291065) q[0];
ry(2.721619902858641) q[1];
rz(0.2552577020732789) q[1];
ry(2.3876680813102262) q[2];
rz(0.6795204257760559) q[2];
ry(-1.8939235410950186) q[3];
rz(0.4572384181882161) q[3];
ry(-0.9428551869085062) q[4];
rz(-0.009293573211580865) q[4];
ry(0.2845140721997854) q[5];
rz(-2.459046533455039) q[5];
ry(-1.4145655040162097) q[6];
rz(1.8140318271552727) q[6];
ry(-2.1370335022099805) q[7];
rz(0.6048671521410967) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.404512044083102) q[0];
rz(-1.8382348601162812) q[0];
ry(-0.16283783303625932) q[1];
rz(-1.6052057961905268) q[1];
ry(0.3912285494096161) q[2];
rz(0.6871036895566789) q[2];
ry(-0.778356925075797) q[3];
rz(1.3478422923009008) q[3];
ry(0.776802254507505) q[4];
rz(-2.809344640883591) q[4];
ry(-1.8783229925922589) q[5];
rz(-2.3737548946015026) q[5];
ry(-0.5584313562864329) q[6];
rz(0.26748459260996044) q[6];
ry(0.7254351161128696) q[7];
rz(0.43823733699691575) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.73233111501536) q[0];
rz(0.8533362106041774) q[0];
ry(1.0923266860883958) q[1];
rz(2.419665403164776) q[1];
ry(-2.150995219558345) q[2];
rz(0.1846994742830894) q[2];
ry(0.29732908349425013) q[3];
rz(-1.308820061249735) q[3];
ry(1.5433804345087987) q[4];
rz(3.000502330790749) q[4];
ry(-2.54533346172953) q[5];
rz(3.03385398458576) q[5];
ry(-2.142492847984438) q[6];
rz(-1.3129173899590025) q[6];
ry(2.0951836285853824) q[7];
rz(-2.2901349670070617) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.9286809821734785) q[0];
rz(-1.2994702743610258) q[0];
ry(0.4755422506687363) q[1];
rz(-0.06664363769760229) q[1];
ry(-0.18451505941746849) q[2];
rz(-1.4236057656700822) q[2];
ry(-0.5965551347583122) q[3];
rz(3.1209306194070834) q[3];
ry(0.13812593064335046) q[4];
rz(2.9189842665753805) q[4];
ry(1.379230150441543) q[5];
rz(-2.6812985360541512) q[5];
ry(-0.7713354785815012) q[6];
rz(-0.0243700475497666) q[6];
ry(-1.2137073084147867) q[7];
rz(1.805954972636107) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.9845566630561953) q[0];
rz(-1.7068050758417534) q[0];
ry(1.6486129655705986) q[1];
rz(-1.164921353562404) q[1];
ry(1.0792779179743395) q[2];
rz(-1.9215298935802059) q[2];
ry(1.4874425022061661) q[3];
rz(-2.324361192919652) q[3];
ry(2.7476018368564237) q[4];
rz(0.5661326575956949) q[4];
ry(-0.2534285819632123) q[5];
rz(-0.39744657751959245) q[5];
ry(-0.8664980122825998) q[6];
rz(-3.1246553166574924) q[6];
ry(-1.5150401742843647) q[7];
rz(-1.6178189979629352) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.8739535689644677) q[0];
rz(-1.3957934722278678) q[0];
ry(-2.7261598633895123) q[1];
rz(2.4890820804085916) q[1];
ry(-2.904628260738388) q[2];
rz(2.5091063816163444) q[2];
ry(-0.8820541026373419) q[3];
rz(-0.9325983924057261) q[3];
ry(1.7868135818025017) q[4];
rz(-1.6723600437857085) q[4];
ry(-1.9384688343213043) q[5];
rz(0.3008419892612769) q[5];
ry(-1.3986273881154574) q[6];
rz(0.9203937475593258) q[6];
ry(2.0689902997769476) q[7];
rz(-0.3872053861417442) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.2303292445319034) q[0];
rz(-2.9918609875230313) q[0];
ry(-1.3694852777064832) q[1];
rz(0.3380150805747335) q[1];
ry(-2.5516350121950246) q[2];
rz(-1.248595265741855) q[2];
ry(-1.7567095522044949) q[3];
rz(-0.5549713121889877) q[3];
ry(1.926041242308366) q[4];
rz(2.2810368998063657) q[4];
ry(-0.39655955144442895) q[5];
rz(-0.862759603701514) q[5];
ry(0.5788688770777979) q[6];
rz(0.42372635317861795) q[6];
ry(1.418654830458686) q[7];
rz(1.0058708708466082) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.9230876137110071) q[0];
rz(0.30847032625386317) q[0];
ry(0.4744389709631215) q[1];
rz(2.3035298809595246) q[1];
ry(-0.9491120958343108) q[2];
rz(0.9121740839106165) q[2];
ry(2.142893142921055) q[3];
rz(-1.1686557002312763) q[3];
ry(3.044872329339151) q[4];
rz(0.3313914200988819) q[4];
ry(0.044638600218997126) q[5];
rz(1.3023057739438997) q[5];
ry(-0.5013060543655021) q[6];
rz(-1.53167393664034) q[6];
ry(-1.9127406248314018) q[7];
rz(-0.7796069264342896) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.6967266927300213) q[0];
rz(0.12065319539980214) q[0];
ry(2.070118890065352) q[1];
rz(-2.9988325295254055) q[1];
ry(-1.381485225644303) q[2];
rz(-2.3074607660544126) q[2];
ry(-0.9382642877138777) q[3];
rz(2.03287302317083) q[3];
ry(0.596537059835282) q[4];
rz(-2.663633979109864) q[4];
ry(2.8220709797491765) q[5];
rz(2.5015738108693095) q[5];
ry(1.3594784109026306) q[6];
rz(-2.7878762898829197) q[6];
ry(2.689020255147844) q[7];
rz(-1.6391921480678615) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.8737210211216966) q[0];
rz(2.4956024261255267) q[0];
ry(0.8553233050683033) q[1];
rz(-3.094446490272886) q[1];
ry(0.06302104254194685) q[2];
rz(-2.6594529717213065) q[2];
ry(1.7828011360449025) q[3];
rz(-2.268426902971317) q[3];
ry(1.9839668207137229) q[4];
rz(-0.45048309804306746) q[4];
ry(-0.2165368740001039) q[5];
rz(1.9675672637382782) q[5];
ry(1.1499828020839982) q[6];
rz(-3.137155822477222) q[6];
ry(-0.7519107928372534) q[7];
rz(1.9548968262328537) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.5449519036648978) q[0];
rz(-2.6239118481906405) q[0];
ry(2.059842124145148) q[1];
rz(-1.453414178491038) q[1];
ry(2.8436897982404714) q[2];
rz(2.2662279447820888) q[2];
ry(1.060050758890351) q[3];
rz(3.0617995725949543) q[3];
ry(-1.3358932473062264) q[4];
rz(1.1371334134325233) q[4];
ry(-2.8117818561984076) q[5];
rz(-0.6566240007162217) q[5];
ry(-0.7904099024282818) q[6];
rz(1.0877185877058624) q[6];
ry(1.8314350641758401) q[7];
rz(-1.0912496811643821) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.5400744240117898) q[0];
rz(-2.305303060801987) q[0];
ry(-0.5258416784699841) q[1];
rz(0.3487316942365809) q[1];
ry(-1.6446093494537872) q[2];
rz(1.7209543540225614) q[2];
ry(-1.3985470642488858) q[3];
rz(2.0605346681967003) q[3];
ry(0.43577475124035486) q[4];
rz(-1.5849180692292146) q[4];
ry(-2.144487597748718) q[5];
rz(0.4456909036415833) q[5];
ry(0.2500502705864198) q[6];
rz(-2.7971843130564173) q[6];
ry(1.3935896892986817) q[7];
rz(-0.17345223314604663) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.9812241087073303) q[0];
rz(-2.098091105779851) q[0];
ry(-1.3063390343089123) q[1];
rz(-0.6516722500933954) q[1];
ry(3.07703563599133) q[2];
rz(-2.178171894574306) q[2];
ry(-1.8455495973250828) q[3];
rz(-0.4881390493390187) q[3];
ry(-2.643216625313974) q[4];
rz(1.734814696998571) q[4];
ry(-2.505928458221951) q[5];
rz(3.1001216800956235) q[5];
ry(-3.05745439912691) q[6];
rz(-1.6238786015056892) q[6];
ry(-1.848249290473034) q[7];
rz(-0.5108267793285971) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.6951133882202742) q[0];
rz(2.3540075158269747) q[0];
ry(0.9941663343992344) q[1];
rz(-3.1261357279631157) q[1];
ry(-0.049269857299695026) q[2];
rz(2.236090655127769) q[2];
ry(-2.325557465350122) q[3];
rz(1.8080015677731192) q[3];
ry(-3.1009723957175654) q[4];
rz(-2.7548605177996666) q[4];
ry(-2.534677884620496) q[5];
rz(3.00583971892043) q[5];
ry(0.4878217566615381) q[6];
rz(-1.7100975657891428) q[6];
ry(1.1537888520829158) q[7];
rz(1.2241360838811195) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.0487463123084244) q[0];
rz(-1.1082127533284432) q[0];
ry(-2.4506056984805404) q[1];
rz(-1.806797509567265) q[1];
ry(2.575139471602642) q[2];
rz(-1.553849981793085) q[2];
ry(-1.551349118936483) q[3];
rz(0.4275184545492897) q[3];
ry(2.5568947455730617) q[4];
rz(-1.9868298768069919) q[4];
ry(-2.4130542950917713) q[5];
rz(-2.847069564287486) q[5];
ry(-1.5599484458437838) q[6];
rz(-1.7450381879645107) q[6];
ry(-2.2528232402525896) q[7];
rz(1.7660336684101168) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.7516217333472728) q[0];
rz(-0.45516756703112726) q[0];
ry(0.9977193680440843) q[1];
rz(-2.1525064147951873) q[1];
ry(-3.1403192213719477) q[2];
rz(-1.5164755901488096) q[2];
ry(0.24999744420108372) q[3];
rz(-2.246933676015397) q[3];
ry(-1.9704328665768402) q[4];
rz(-1.0418511717595251) q[4];
ry(-2.2966301024028692) q[5];
rz(-0.5019015039231435) q[5];
ry(0.9211572996101437) q[6];
rz(-3.1281752069592437) q[6];
ry(-2.800486589852349) q[7];
rz(0.9832618973382766) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.6093814343066901) q[0];
rz(2.2746969490847775) q[0];
ry(-1.7683498000336824) q[1];
rz(1.1159856306351186) q[1];
ry(-2.654984007700265) q[2];
rz(-2.8567373896733645) q[2];
ry(0.8643909939204811) q[3];
rz(-2.5078103302375614) q[3];
ry(1.8771848803926003) q[4];
rz(-1.5611660496054167) q[4];
ry(0.7053635305494667) q[5];
rz(1.4719910720449376) q[5];
ry(-0.20326452987939553) q[6];
rz(-1.7343889090115305) q[6];
ry(0.02946735887949714) q[7];
rz(-1.3293725212983636) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.8821410190579054) q[0];
rz(1.3739637828413336) q[0];
ry(2.650854232496472) q[1];
rz(2.482573713663701) q[1];
ry(-0.7884264666108405) q[2];
rz(0.8872754957536699) q[2];
ry(1.4987582485064892) q[3];
rz(-2.210458804675301) q[3];
ry(0.3420315794222635) q[4];
rz(-2.8515225095715526) q[4];
ry(2.7351544731668853) q[5];
rz(-1.8208802789990255) q[5];
ry(-0.39639084064645075) q[6];
rz(0.7429775438785101) q[6];
ry(-1.0792382943148215) q[7];
rz(1.4427439583083501) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.9441641417219424) q[0];
rz(-0.8223247309261453) q[0];
ry(0.23934062548785115) q[1];
rz(1.1548210812615054) q[1];
ry(2.6359670066456045) q[2];
rz(2.561238551188857) q[2];
ry(-1.412376353608634) q[3];
rz(-0.05699090765225368) q[3];
ry(1.6411810264477777) q[4];
rz(2.8897208859510646) q[4];
ry(0.7801632205522387) q[5];
rz(-2.592776997467161) q[5];
ry(-2.925862140709872) q[6];
rz(-1.6200223098837352) q[6];
ry(-2.474747951306038) q[7];
rz(-2.5386107693164357) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.271155809517258) q[0];
rz(-1.2771143768532554) q[0];
ry(2.6383533352350232) q[1];
rz(0.142926313327691) q[1];
ry(1.2549873823259243) q[2];
rz(-1.1897529277463494) q[2];
ry(-2.9662750731933123) q[3];
rz(-0.2829948664720158) q[3];
ry(1.1912016625830593) q[4];
rz(3.116750543181836) q[4];
ry(1.3809089953149298) q[5];
rz(-2.218016701632849) q[5];
ry(1.3393696124063301) q[6];
rz(-1.2545586126550525) q[6];
ry(-1.2486113296162835) q[7];
rz(2.937196704635369) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.168058347736846) q[0];
rz(0.9969693984568222) q[0];
ry(-0.1533222373425484) q[1];
rz(-2.9729019187689536) q[1];
ry(2.0091280269738) q[2];
rz(2.4380371900116176) q[2];
ry(0.1258277515454859) q[3];
rz(0.5128499305813206) q[3];
ry(-1.5388863832640114) q[4];
rz(-1.354139596823404) q[4];
ry(0.4087838195808491) q[5];
rz(-0.16245057954752085) q[5];
ry(0.1009529557355479) q[6];
rz(-1.4063473132575846) q[6];
ry(2.304508379168163) q[7];
rz(-1.340844959696411) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8209421670854882) q[0];
rz(-0.07318377234821584) q[0];
ry(0.46094121241187835) q[1];
rz(2.9856387423765525) q[1];
ry(0.15271660755278174) q[2];
rz(1.1107562425618953) q[2];
ry(-0.051857952862482716) q[3];
rz(-2.866763696350311) q[3];
ry(1.4588291467010155) q[4];
rz(-0.3407766198786745) q[4];
ry(-1.1856214172018822) q[5];
rz(-2.144691803043547) q[5];
ry(2.0945816851391883) q[6];
rz(-1.5941847445972723) q[6];
ry(2.124540994384332) q[7];
rz(-1.9252159721781226) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.0873088644504632) q[0];
rz(1.6226073754320178) q[0];
ry(-2.2519286351798047) q[1];
rz(-0.973347294197044) q[1];
ry(2.152011064909511) q[2];
rz(-0.6311002262089449) q[2];
ry(1.7188317873598375) q[3];
rz(2.2282313855079754) q[3];
ry(-1.07088526864288) q[4];
rz(-2.048887170734282) q[4];
ry(0.0986369624629022) q[5];
rz(-2.2782481006044835) q[5];
ry(3.087274015514251) q[6];
rz(1.698790578012526) q[6];
ry(-2.66727546466482) q[7];
rz(-3.049082399254978) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.057313014711121) q[0];
rz(1.0419279927648883) q[0];
ry(0.12795442040753516) q[1];
rz(1.6240775401132117) q[1];
ry(0.09851208059030991) q[2];
rz(0.6851162599219532) q[2];
ry(-2.126750735277204) q[3];
rz(-1.6370200011307325) q[3];
ry(-0.6382975381909279) q[4];
rz(1.9377138987750602) q[4];
ry(1.3240249608513768) q[5];
rz(-1.817259569123889) q[5];
ry(-1.8486337741338792) q[6];
rz(-0.6035483694938898) q[6];
ry(-1.073504403989532) q[7];
rz(0.5966575636819271) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.6809108447037409) q[0];
rz(0.9506463775791922) q[0];
ry(8.293392109948172e-06) q[1];
rz(2.51247630242291) q[1];
ry(1.7042221636004493) q[2];
rz(-2.8158598375102506) q[2];
ry(1.8926275901097318) q[3];
rz(-0.12797460501064606) q[3];
ry(1.9159400545614436) q[4];
rz(1.5593562457891323) q[4];
ry(-1.5487280199451348) q[5];
rz(-1.0020248878133855) q[5];
ry(3.0994782856205885) q[6];
rz(-0.1489063021702952) q[6];
ry(0.9184741700650392) q[7];
rz(-0.3633812852189493) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.1162320208396106) q[0];
rz(-2.72355693136615) q[0];
ry(-1.1873481818604628) q[1];
rz(-3.092758283344378) q[1];
ry(3.0842225156708105) q[2];
rz(-2.1259033383971726) q[2];
ry(3.1037181267847647) q[3];
rz(-1.6156710387242628) q[3];
ry(2.1834635395427755) q[4];
rz(-3.0326193898686884) q[4];
ry(0.27695524881967426) q[5];
rz(-2.9838785782701356) q[5];
ry(-1.6121506045122072) q[6];
rz(1.0821961641702238) q[6];
ry(-1.0915932853080408) q[7];
rz(2.3480854483633937) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.7986037843900133) q[0];
rz(-0.1911949779616495) q[0];
ry(-1.7667287134322782) q[1];
rz(1.5674060595130725) q[1];
ry(-2.8898606682361168) q[2];
rz(-1.5469214783377319) q[2];
ry(0.6110825014729043) q[3];
rz(-2.5391480823653594) q[3];
ry(1.3591711621688178) q[4];
rz(-1.280043904144409) q[4];
ry(1.6372611260670296) q[5];
rz(2.6609229917189294) q[5];
ry(-0.2692388768107019) q[6];
rz(-1.2307531781430319) q[6];
ry(1.6576778187431562) q[7];
rz(1.6496715457440763) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.6163702502316764) q[0];
rz(3.0728509805643336) q[0];
ry(-1.5550149371933308) q[1];
rz(0.7790946326739226) q[1];
ry(-3.073328304294248) q[2];
rz(-2.1369201926042183) q[2];
ry(0.029934488738181848) q[3];
rz(-0.6479927457539354) q[3];
ry(-0.010796976115666546) q[4];
rz(0.7544168095429241) q[4];
ry(0.09895972236904971) q[5];
rz(2.063816859460174) q[5];
ry(1.553616215957395) q[6];
rz(-1.5543929371124758) q[6];
ry(1.5426649993004058) q[7];
rz(-1.8460437127165514) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.6429668816414045) q[0];
rz(-2.6240447047570483) q[0];
ry(-0.07323079957037049) q[1];
rz(-2.1033569417327267) q[1];
ry(1.576177610725097) q[2];
rz(0.528451770473296) q[2];
ry(2.594687408069338) q[3];
rz(3.116578859345987) q[3];
ry(0.4017954473473223) q[4];
rz(-0.22267318694930704) q[4];
ry(1.5067187956917318) q[5];
rz(0.5904412608707799) q[5];
ry(1.5908037187941577) q[6];
rz(1.7178292938908981) q[6];
ry(-0.00765538228906415) q[7];
rz(-1.7963560536825929) q[7];