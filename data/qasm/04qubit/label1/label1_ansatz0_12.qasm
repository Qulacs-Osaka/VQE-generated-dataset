OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.05499144122457585) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.22085463482369355) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.08781488950642365) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.12957207122359618) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.32719810150650425) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.04777210675090311) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06880115479936527) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.07932903354115356) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.018583272265299572) q[3];
cx q[2],q[3];
rz(-0.10594284324697835) q[0];
rz(-0.04472590146747393) q[1];
rz(-0.03037003105547425) q[2];
rz(-0.06197781714235304) q[3];
rx(-0.3143320521231827) q[0];
rx(-0.18361599844461332) q[1];
rx(-0.14792342004515358) q[2];
rx(-0.14285656413405304) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.08544414835831982) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17166435774066335) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.012411414954644052) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.15972494575280954) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.2446396890631791) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09590756218961353) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.004009127240213969) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11756916877146573) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.007563750485621411) q[3];
cx q[2],q[3];
rz(-0.05742353800599092) q[0];
rz(-0.1357216603767459) q[1];
rz(-0.004763706339011283) q[2];
rz(-0.022830708210802925) q[3];
rx(-0.33434855790918555) q[0];
rx(-0.14710342055276923) q[1];
rx(-0.26432755831925997) q[2];
rx(-0.20960661346176407) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03536790303200135) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12446992932422109) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.11531361717304814) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.06283544903473251) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.09663548255886528) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.04418541783232463) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.03296348278400954) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.17799708988411367) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08733870799799812) q[3];
cx q[2],q[3];
rz(0.00843164337766453) q[0];
rz(-0.1417946653316038) q[1];
rz(0.04056279314953757) q[2];
rz(-0.016772957774668314) q[3];
rx(-0.29688421158393263) q[0];
rx(-0.06018540554577924) q[1];
rx(-0.3318871839937182) q[2];
rx(-0.18422047611750916) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.07111766762912473) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12325089425233202) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.16024684024868652) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.12194805071465842) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.018156115404714224) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.06817099474997824) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0950353818972694) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.20728560957339004) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11285986093621347) q[3];
cx q[2],q[3];
rz(0.024643600539877894) q[0];
rz(-0.0751796172799905) q[1];
rz(0.057929911286916165) q[2];
rz(0.004601792393030221) q[3];
rx(-0.2342983878120425) q[0];
rx(-0.10405059728490257) q[1];
rx(-0.3490982369349988) q[2];
rx(-0.13404989710332563) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.13204858522739435) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.16308390806748257) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.132543442704153) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2775665990313424) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.11082689521133542) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.012805941390666603) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.1408118784237846) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.17592944997643858) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07728062055933595) q[3];
cx q[2],q[3];
rz(0.04675170099549769) q[0];
rz(0.0377109008748498) q[1];
rz(0.06537366315896796) q[2];
rz(0.011102497046468877) q[3];
rx(-0.2516046864660684) q[0];
rx(-0.17831687356976603) q[1];
rx(-0.4017015562495495) q[2];
rx(-0.12835988790412547) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.009707842636835122) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1598498226684653) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.04582296335039851) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.22908042816525387) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.07721385083786213) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.049634162294351976) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2800553612804221) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.18905593720194133) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.035862726535898566) q[3];
cx q[2],q[3];
rz(0.06557268009497746) q[0];
rz(0.028520037520692775) q[1];
rz(-0.061532577813388084) q[2];
rz(0.08741214701601124) q[3];
rx(-0.2056959963037023) q[0];
rx(-0.19613909858220832) q[1];
rx(-0.3775930621822857) q[2];
rx(-0.12965523201711654) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1086613417649181) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1967177651603924) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.03357789985777627) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.30414744470007) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.02071226966905781) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13066447460446495) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2976537421505954) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1732983091508435) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.07918451987208051) q[3];
cx q[2],q[3];
rz(0.02752817898238123) q[0];
rz(0.14934015974532378) q[1];
rz(-0.16714328973442955) q[2];
rz(0.029667613763040786) q[3];
rx(-0.24694638333968935) q[0];
rx(-0.23278996384031075) q[1];
rx(-0.37014640056715453) q[2];
rx(-0.12380673689281045) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.019131908093531828) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.16460972442833388) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.15709996291231454) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2933560459605908) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.05753113366722513) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.1828111168285526) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2666288608079778) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.19901891522289203) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.20139444801605488) q[3];
cx q[2],q[3];
rz(0.028947193278551865) q[0];
rz(0.12232590121963603) q[1];
rz(-0.1963192202696334) q[2];
rz(0.03635330035077705) q[3];
rx(-0.23103331745071196) q[0];
rx(-0.24982406563842016) q[1];
rx(-0.43797977296090523) q[2];
rx(-0.1319000815390929) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.004874699209988397) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.04126256204335263) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05377923713747793) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2851519573569462) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.0061210705858138256) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08173392772910593) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.2595435628777172) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.12021536589673674) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2480053480683118) q[3];
cx q[2],q[3];
rz(0.09167811560237261) q[0];
rz(0.12130052112430738) q[1];
rz(-0.33602289926876755) q[2];
rz(-0.052298731375302825) q[3];
rx(-0.25188129710309975) q[0];
rx(-0.17463850564145766) q[1];
rx(-0.5451851367981174) q[2];
rx(-0.19118517967967008) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.03943589984034297) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.03335991939787255) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04336401862761591) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.21150246037723147) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.12190341558309967) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.17358586357027808) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.06377288676928472) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.15505699636116865) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2426857283129674) q[3];
cx q[2],q[3];
rz(0.01175075694123388) q[0];
rz(0.09542589997235622) q[1];
rz(-0.30929312791622726) q[2];
rz(-0.1391685336883748) q[3];
rx(-0.27952409064157474) q[0];
rx(-0.18727250955054298) q[1];
rx(-0.6225114503493847) q[2];
rx(-0.2213998473256318) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0026268047924783502) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.011495123367357543) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.030556895968157402) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.13552275897025032) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.1536232819938548) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.19561719518271098) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.09224608686766517) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.11572653966597489) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2965230658940502) q[3];
cx q[2],q[3];
rz(-0.03380564038305504) q[0];
rz(-0.008065049116288792) q[1];
rz(-0.14664699824835395) q[2];
rz(-0.19935832187269154) q[3];
rx(-0.18557525672354552) q[0];
rx(-0.23325096160734457) q[1];
rx(-0.6017775886343943) q[2];
rx(-0.23706209522233132) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.11943958738574241) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0991700732380093) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04207795089928797) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.10276860438220128) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.19914376738140493) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.03674887361258044) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.24195540001386362) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.1753074025175544) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.29274975456666996) q[3];
cx q[2],q[3];
rz(0.032931499034303965) q[0];
rz(-0.2543560227451574) q[1];
rz(-0.08282285003312785) q[2];
rz(-0.19399327365206862) q[3];
rx(-0.15386462837560713) q[0];
rx(-0.1942054312685357) q[1];
rx(-0.6401598381698422) q[2];
rx(-0.20302198056799017) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.09309125064167499) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0026655976725164835) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.01596607929786044) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.12243746846445924) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.26398663142981393) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.014013873239889804) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.31610932758278343) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.2503739487846904) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.2302007240511515) q[3];
cx q[2],q[3];
rz(0.023237828287634196) q[0];
rz(-0.22412101335872012) q[1];
rz(-0.06473230418158085) q[2];
rz(-0.04994490120355259) q[3];
rx(-0.039993038478076504) q[0];
rx(-0.36074316835963144) q[1];
rx(-0.6466420039342307) q[2];
rx(-0.13138493459779793) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.22165051367858005) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.0013843332818020398) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05482175698164406) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2211243799298725) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.08008188333502034) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.14651211018832364) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.33784948728484715) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.02104053355716192) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17264661871047515) q[3];
cx q[2],q[3];
rz(0.16272254712677323) q[0];
rz(-0.23075577661384394) q[1];
rz(-0.12308183588886529) q[2];
rz(-0.02865110766983916) q[3];
rx(0.029832866656386334) q[0];
rx(-0.3221705548295499) q[1];
rx(-0.6399934469562941) q[2];
rx(-0.1408570119125185) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.056596493887189736) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.01089931920418146) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05875061459133532) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.462049759437195) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.1636037450692131) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.14191067539685068) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.34357244929764263) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.31080207383495145) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08926001597479306) q[3];
cx q[2],q[3];
rz(0.14833860356357184) q[0];
rz(-0.25415069508459126) q[1];
rz(0.1999468105122811) q[2];
rz(-0.06179692881849224) q[3];
rx(0.05416944659361507) q[0];
rx(-0.25784866053312616) q[1];
rx(-0.2664520731307964) q[2];
rx(-0.35636184936759013) q[3];