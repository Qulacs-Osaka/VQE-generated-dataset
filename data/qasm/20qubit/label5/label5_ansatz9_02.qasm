OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.10881484270049216) q[0];
ry(-0.23162091540900856) q[1];
cx q[0],q[1];
ry(0.007426817970191253) q[0];
ry(-3.0529555769166272) q[1];
cx q[0],q[1];
ry(-0.2622474301627449) q[2];
ry(-2.853240029630407) q[3];
cx q[2],q[3];
ry(-2.999896340790479) q[2];
ry(-2.7554094221595813) q[3];
cx q[2],q[3];
ry(1.9010253890592266) q[4];
ry(-1.5185915726814327) q[5];
cx q[4],q[5];
ry(1.9270610425549857) q[4];
ry(2.8864836319908704) q[5];
cx q[4],q[5];
ry(-1.3021875193717687) q[6];
ry(0.506406015274825) q[7];
cx q[6],q[7];
ry(1.8553867821385062) q[6];
ry(-2.6406877882445374) q[7];
cx q[6],q[7];
ry(-0.2967597307134726) q[8];
ry(-0.998998887605041) q[9];
cx q[8],q[9];
ry(2.1412661129691632) q[8];
ry(2.894958226750234) q[9];
cx q[8],q[9];
ry(-0.34100508193003365) q[10];
ry(2.1118731366823766) q[11];
cx q[10],q[11];
ry(3.0345829482467903) q[10];
ry(2.9903437383259073) q[11];
cx q[10],q[11];
ry(-0.027646822437855673) q[12];
ry(-2.5266344908671345) q[13];
cx q[12],q[13];
ry(0.6035796560335701) q[12];
ry(0.931848352551037) q[13];
cx q[12],q[13];
ry(1.7754743405753421) q[14];
ry(0.5375264592276014) q[15];
cx q[14],q[15];
ry(2.145548661910394) q[14];
ry(-0.8591411007132521) q[15];
cx q[14],q[15];
ry(2.307882860182579) q[16];
ry(1.112127593830869) q[17];
cx q[16],q[17];
ry(-2.2389609717197096) q[16];
ry(-1.5259416785663218) q[17];
cx q[16],q[17];
ry(-1.4241952871671586) q[18];
ry(-0.5314237916918475) q[19];
cx q[18],q[19];
ry(-2.6195380511161397) q[18];
ry(3.011179243955791) q[19];
cx q[18],q[19];
ry(-1.674303910070912) q[0];
ry(1.9293064649277043) q[2];
cx q[0],q[2];
ry(-2.063509843005124) q[0];
ry(2.442011762709517) q[2];
cx q[0],q[2];
ry(1.0007643384306737) q[2];
ry(0.5962897149295054) q[4];
cx q[2],q[4];
ry(-2.141964162837715) q[2];
ry(-2.26096645200423) q[4];
cx q[2],q[4];
ry(-1.181789912182234) q[4];
ry(0.7585416656181199) q[6];
cx q[4],q[6];
ry(0.87720699799518) q[4];
ry(-0.638818545739124) q[6];
cx q[4],q[6];
ry(-2.1400958091510933) q[6];
ry(2.653263392817475) q[8];
cx q[6],q[8];
ry(-0.008149914614500098) q[6];
ry(-3.1159799199306732) q[8];
cx q[6],q[8];
ry(0.039134414450285426) q[8];
ry(0.21982474742629865) q[10];
cx q[8],q[10];
ry(3.131248180944905) q[8];
ry(-2.19672047065115) q[10];
cx q[8],q[10];
ry(-2.9721743272524135) q[10];
ry(0.5775995470256561) q[12];
cx q[10],q[12];
ry(1.5563307389466907) q[10];
ry(1.5760077284146081) q[12];
cx q[10],q[12];
ry(0.6124608522638475) q[12];
ry(-0.8106121102438931) q[14];
cx q[12],q[14];
ry(-1.2306753223031093) q[12];
ry(-2.6133087920049674) q[14];
cx q[12],q[14];
ry(-1.219887448224419) q[14];
ry(-0.3665311680933491) q[16];
cx q[14],q[16];
ry(-1.5759343569174333) q[14];
ry(-1.566245442830568) q[16];
cx q[14],q[16];
ry(1.5223273307546046) q[16];
ry(-0.03617516044779023) q[18];
cx q[16],q[18];
ry(2.499974816526791) q[16];
ry(-2.419421013696655) q[18];
cx q[16],q[18];
ry(0.21905735054858422) q[1];
ry(-2.2437174987293824) q[3];
cx q[1],q[3];
ry(0.17660859058229397) q[1];
ry(2.2190748536689866) q[3];
cx q[1],q[3];
ry(2.83846069803039) q[3];
ry(-0.03457228548795952) q[5];
cx q[3],q[5];
ry(-1.5437115269655477) q[3];
ry(-2.815374277414716) q[5];
cx q[3],q[5];
ry(2.94205313000204) q[5];
ry(3.05886712129873) q[7];
cx q[5],q[7];
ry(1.5143466463776363) q[5];
ry(1.6353784549716945) q[7];
cx q[5],q[7];
ry(0.06395488309436412) q[7];
ry(1.0678739557629287) q[9];
cx q[7],q[9];
ry(-0.006229314397757868) q[7];
ry(2.453197890448031) q[9];
cx q[7],q[9];
ry(-1.8976387072056715) q[9];
ry(-2.490145337451055) q[11];
cx q[9],q[11];
ry(-0.4821064924595515) q[9];
ry(0.003867222744492693) q[11];
cx q[9],q[11];
ry(1.3957263263052395) q[11];
ry(-0.12452904229278101) q[13];
cx q[11],q[13];
ry(0.7087550797878417) q[11];
ry(-2.9272412922204682) q[13];
cx q[11],q[13];
ry(-0.9874484151229748) q[13];
ry(0.4567450250141958) q[15];
cx q[13],q[15];
ry(-1.704415362576544) q[13];
ry(-0.9513906592609453) q[15];
cx q[13],q[15];
ry(-1.411866906109137) q[15];
ry(1.1880584414063564) q[17];
cx q[15],q[17];
ry(-3.090896530563885) q[15];
ry(3.138942189660214) q[17];
cx q[15],q[17];
ry(0.6037929333150636) q[17];
ry(1.744840398612907) q[19];
cx q[17],q[19];
ry(-1.0677127837942129) q[17];
ry(2.479074521457174) q[19];
cx q[17],q[19];
ry(-2.6058997567699898) q[0];
ry(2.5867565981326406) q[3];
cx q[0],q[3];
ry(-0.011786658849054277) q[0];
ry(0.29333044014682785) q[3];
cx q[0],q[3];
ry(-2.865131655415457) q[1];
ry(-1.2725218534698817) q[2];
cx q[1],q[2];
ry(-0.03805907780326985) q[1];
ry(2.661663771942751) q[2];
cx q[1],q[2];
ry(-1.8594121922845117) q[2];
ry(1.3602611250648105) q[5];
cx q[2],q[5];
ry(-2.043567862038323) q[2];
ry(1.3230146281507587) q[5];
cx q[2],q[5];
ry(2.7023969801186967) q[3];
ry(-2.9092117743679258) q[4];
cx q[3],q[4];
ry(1.0411254883880847) q[3];
ry(-0.33187084734173045) q[4];
cx q[3],q[4];
ry(0.7942660226067898) q[4];
ry(-1.516766074523085) q[7];
cx q[4],q[7];
ry(0.42771416752033353) q[4];
ry(0.023518745806446122) q[7];
cx q[4],q[7];
ry(1.2632830973817615) q[5];
ry(-0.837708047259412) q[6];
cx q[5],q[6];
ry(2.228541284421131) q[5];
ry(-2.0179366898848965) q[6];
cx q[5],q[6];
ry(2.3363527141779175) q[6];
ry(2.954741651909074) q[9];
cx q[6],q[9];
ry(-3.0579486768820754) q[6];
ry(-2.574900019682064) q[9];
cx q[6],q[9];
ry(0.4710928744011582) q[7];
ry(-1.0749636842798203) q[8];
cx q[7],q[8];
ry(0.02926015398642061) q[7];
ry(3.0928727033017274) q[8];
cx q[7],q[8];
ry(-1.6427561967081772) q[8];
ry(0.20602846520056417) q[11];
cx q[8],q[11];
ry(3.1274376272282614) q[8];
ry(0.005688845311049209) q[11];
cx q[8],q[11];
ry(-1.3557506221695386) q[9];
ry(1.14924932543149) q[10];
cx q[9],q[10];
ry(-1.8039488009830407) q[9];
ry(0.17796618905009007) q[10];
cx q[9],q[10];
ry(-1.1850006858229145) q[10];
ry(-1.441556231332727) q[13];
cx q[10],q[13];
ry(-2.257486954669794) q[10];
ry(0.35738618868311267) q[13];
cx q[10],q[13];
ry(2.4862047647685475) q[11];
ry(1.2629423505812616) q[12];
cx q[11],q[12];
ry(-3.1404457028219244) q[11];
ry(-3.1400808537053457) q[12];
cx q[11],q[12];
ry(-0.5129594818963599) q[12];
ry(2.145979590804176) q[15];
cx q[12],q[15];
ry(-0.005542678298205317) q[12];
ry(3.13943232953325) q[15];
cx q[12],q[15];
ry(0.1302152593654995) q[13];
ry(-1.047692542884739) q[14];
cx q[13],q[14];
ry(0.030048889933631067) q[13];
ry(0.22759349218988412) q[14];
cx q[13],q[14];
ry(-2.519937332723653) q[14];
ry(-0.5225510417245873) q[17];
cx q[14],q[17];
ry(3.074458842200272) q[14];
ry(0.030857144846180514) q[17];
cx q[14],q[17];
ry(-1.072894904169913) q[15];
ry(1.2514378477892327) q[16];
cx q[15],q[16];
ry(-0.016489622337062748) q[15];
ry(0.00033420626456053546) q[16];
cx q[15],q[16];
ry(2.364746869788246) q[16];
ry(-2.201733040404977) q[19];
cx q[16],q[19];
ry(3.0514356201917026) q[16];
ry(1.0865851750900462) q[19];
cx q[16],q[19];
ry(-1.931718747260002) q[17];
ry(2.936854954971424) q[18];
cx q[17],q[18];
ry(0.4344236268250808) q[17];
ry(2.788492975377143) q[18];
cx q[17],q[18];
ry(1.6205692111538506) q[0];
ry(2.6224566401186196) q[1];
cx q[0],q[1];
ry(3.0214312860626333) q[0];
ry(-0.09839861052084123) q[1];
cx q[0],q[1];
ry(-2.2422274175067596) q[2];
ry(-1.6350789432424964) q[3];
cx q[2],q[3];
ry(0.13533084091293723) q[2];
ry(-1.1977559678055891) q[3];
cx q[2],q[3];
ry(-0.8008182834546513) q[4];
ry(2.849753151066217) q[5];
cx q[4],q[5];
ry(2.8176378051667355) q[4];
ry(2.1730263274105863) q[5];
cx q[4],q[5];
ry(2.514643840766629) q[6];
ry(-1.0911271039963981) q[7];
cx q[6],q[7];
ry(1.6684017570111198) q[6];
ry(-1.6336452553782062) q[7];
cx q[6],q[7];
ry(2.950860272413645) q[8];
ry(-2.9700613893960757) q[9];
cx q[8],q[9];
ry(1.5314973250294341) q[8];
ry(1.2321738854593467) q[9];
cx q[8],q[9];
ry(-0.5819405468153462) q[10];
ry(-3.095149220531052) q[11];
cx q[10],q[11];
ry(1.2719644379657522) q[10];
ry(2.120370184914977) q[11];
cx q[10],q[11];
ry(0.7865565608390979) q[12];
ry(-1.3013349392802915) q[13];
cx q[12],q[13];
ry(-0.0028307655867023224) q[12];
ry(0.0018652773441386375) q[13];
cx q[12],q[13];
ry(-0.02458045177051223) q[14];
ry(0.4981685944145758) q[15];
cx q[14],q[15];
ry(1.7829502439246143) q[14];
ry(1.4900360740149932) q[15];
cx q[14],q[15];
ry(0.6828568088698228) q[16];
ry(1.807999799738102) q[17];
cx q[16],q[17];
ry(-2.821117269107871) q[16];
ry(-1.022412027606477) q[17];
cx q[16],q[17];
ry(-0.17053824971113407) q[18];
ry(1.0122713302336717) q[19];
cx q[18],q[19];
ry(-1.3605937532102954) q[18];
ry(0.5575199245310811) q[19];
cx q[18],q[19];
ry(1.5618921841333797) q[0];
ry(-0.10622699100707944) q[2];
cx q[0],q[2];
ry(0.447864172855661) q[0];
ry(1.130496144882395) q[2];
cx q[0],q[2];
ry(2.307225090551037) q[2];
ry(0.11305844690065794) q[4];
cx q[2],q[4];
ry(2.973851785138947) q[2];
ry(0.019569209859149353) q[4];
cx q[2],q[4];
ry(0.5075443385786684) q[4];
ry(2.776770931560956) q[6];
cx q[4],q[6];
ry(-2.894836563345792) q[4];
ry(2.4390236633344866) q[6];
cx q[4],q[6];
ry(-2.8413301372366604) q[6];
ry(2.362028042840379) q[8];
cx q[6],q[8];
ry(-0.006887182093195499) q[6];
ry(-0.011808689790711924) q[8];
cx q[6],q[8];
ry(1.9749866851977433) q[8];
ry(1.5938985852446679) q[10];
cx q[8],q[10];
ry(1.5794454985067496) q[8];
ry(0.025271175819486465) q[10];
cx q[8],q[10];
ry(-0.004089707034528267) q[10];
ry(-0.035669375269099746) q[12];
cx q[10],q[12];
ry(-1.5722448393333974) q[10];
ry(1.5728305972179388) q[12];
cx q[10],q[12];
ry(-0.042801064940353156) q[12];
ry(0.0002734029521607416) q[14];
cx q[12],q[14];
ry(-3.1150240888816976) q[12];
ry(-1.0976726678112678) q[14];
cx q[12],q[14];
ry(-1.3294538186989766) q[14];
ry(1.605030090661418) q[16];
cx q[14],q[16];
ry(-3.136822555895187) q[14];
ry(0.004781818257129089) q[16];
cx q[14],q[16];
ry(-1.5048310340799589) q[16];
ry(1.8441587793325507) q[18];
cx q[16],q[18];
ry(2.9766417774324414) q[16];
ry(2.9090085117582234) q[18];
cx q[16],q[18];
ry(-2.1327415849481532) q[1];
ry(-0.21728285567108727) q[3];
cx q[1],q[3];
ry(-1.5555304769254674) q[1];
ry(1.5595600799712956) q[3];
cx q[1],q[3];
ry(1.5505965724388686) q[3];
ry(-0.4435121322104277) q[5];
cx q[3],q[5];
ry(-1.6096546932135283) q[3];
ry(-1.5345560246187864) q[5];
cx q[3],q[5];
ry(-1.57062245610018) q[5];
ry(2.8914911758061455) q[7];
cx q[5],q[7];
ry(1.580302182055059) q[5];
ry(1.5683379772141364) q[7];
cx q[5],q[7];
ry(-1.0033798065090793) q[7];
ry(-3.1073024890882466) q[9];
cx q[7],q[9];
ry(1.563546386299653) q[7];
ry(3.1189186807961637) q[9];
cx q[7],q[9];
ry(0.9117710695612526) q[9];
ry(-0.8057558213758131) q[11];
cx q[9],q[11];
ry(0.0005861128200432704) q[9];
ry(-3.1407894249604857) q[11];
cx q[9],q[11];
ry(-0.421552139210565) q[11];
ry(-0.44432757836733183) q[13];
cx q[11],q[13];
ry(-1.62677425518244) q[11];
ry(1.4640420999335022) q[13];
cx q[11],q[13];
ry(3.079708064556992) q[13];
ry(0.9381656027077483) q[15];
cx q[13],q[15];
ry(-0.22130102245058705) q[13];
ry(-2.952982693826713) q[15];
cx q[13],q[15];
ry(1.1165901117610817) q[15];
ry(2.182448902949595) q[17];
cx q[15],q[17];
ry(-0.003772214523247133) q[15];
ry(-3.1378692144757903) q[17];
cx q[15],q[17];
ry(-1.5559133331466022) q[17];
ry(-0.7132643290492426) q[19];
cx q[17],q[19];
ry(-0.41909851960853806) q[17];
ry(-0.09745031656164471) q[19];
cx q[17],q[19];
ry(0.38648550850376273) q[0];
ry(0.7609094414524976) q[3];
cx q[0],q[3];
ry(-3.056842304865205) q[0];
ry(-1.8356969181879093) q[3];
cx q[0],q[3];
ry(-2.944484205730299) q[1];
ry(-0.09762900465660719) q[2];
cx q[1],q[2];
ry(0.2861771336672163) q[1];
ry(1.0175146643037118) q[2];
cx q[1],q[2];
ry(-2.0154484064424256) q[2];
ry(-3.077623236332291) q[5];
cx q[2],q[5];
ry(3.0199879079056675) q[2];
ry(-0.12877895224561087) q[5];
cx q[2],q[5];
ry(1.7247164143673288) q[3];
ry(-1.2477958065079475) q[4];
cx q[3],q[4];
ry(-2.301841165829157) q[3];
ry(-2.1936734783495773) q[4];
cx q[3],q[4];
ry(-1.1365607808606635) q[4];
ry(0.7433090124642294) q[7];
cx q[4],q[7];
ry(-0.003055410765746478) q[4];
ry(-0.026668550367921817) q[7];
cx q[4],q[7];
ry(2.786598359782018) q[5];
ry(1.0036679157259252) q[6];
cx q[5],q[6];
ry(-0.02257498003105433) q[5];
ry(-0.037663581747237274) q[6];
cx q[5],q[6];
ry(-3.0292311128152734) q[6];
ry(2.771359905252173) q[9];
cx q[6],q[9];
ry(-0.012164828311982411) q[6];
ry(2.2780517140958656) q[9];
cx q[6],q[9];
ry(-1.7486193361294236) q[7];
ry(-1.2903583643586156) q[8];
cx q[7],q[8];
ry(-1.5746038237113007) q[7];
ry(1.566299233063493) q[8];
cx q[7],q[8];
ry(-2.9540415125903237) q[8];
ry(2.7006203996775815) q[11];
cx q[8],q[11];
ry(3.140152374531933) q[8];
ry(-0.0044191935571058485) q[11];
cx q[8],q[11];
ry(0.5578097175675528) q[9];
ry(3.1009001395880342) q[10];
cx q[9],q[10];
ry(0.06790977144840138) q[9];
ry(1.5386821743315569) q[10];
cx q[9],q[10];
ry(-1.5030055575398782) q[10];
ry(-0.8105637766775305) q[13];
cx q[10],q[13];
ry(3.1412337611358367) q[10];
ry(3.1403762775249784) q[13];
cx q[10],q[13];
ry(-0.40184070263997107) q[11];
ry(1.6294698012939417) q[12];
cx q[11],q[12];
ry(-1.660660538677514) q[11];
ry(1.5209514930803092) q[12];
cx q[11],q[12];
ry(3.112217833025574) q[12];
ry(0.942338510672996) q[15];
cx q[12],q[15];
ry(-3.111036654101643) q[12];
ry(-0.735176937574138) q[15];
cx q[12],q[15];
ry(2.4421071127967617) q[13];
ry(-0.8686258813330466) q[14];
cx q[13],q[14];
ry(0.06619024218053848) q[13];
ry(1.596453054175569) q[14];
cx q[13],q[14];
ry(-0.1433283864354511) q[14];
ry(0.2298493671070406) q[17];
cx q[14],q[17];
ry(3.1381501880373577) q[14];
ry(3.138498971548212) q[17];
cx q[14],q[17];
ry(-0.2552934147621668) q[15];
ry(2.1669591380916975) q[16];
cx q[15],q[16];
ry(0.0010049258188882303) q[15];
ry(-0.004209771717368582) q[16];
cx q[15],q[16];
ry(-0.773162165255143) q[16];
ry(-1.4929053729936221) q[19];
cx q[16],q[19];
ry(1.7628968703800607) q[16];
ry(0.11991930273280012) q[19];
cx q[16],q[19];
ry(1.4081628742756815) q[17];
ry(0.23255707255088634) q[18];
cx q[17],q[18];
ry(-3.1198872413110914) q[17];
ry(2.9373523749712622) q[18];
cx q[17],q[18];
ry(1.3616576583099647) q[0];
ry(-0.6053405874743105) q[1];
cx q[0],q[1];
ry(-1.726232091214256) q[0];
ry(1.1443444259484776) q[1];
cx q[0],q[1];
ry(-0.4979502691846056) q[2];
ry(1.558082937570715) q[3];
cx q[2],q[3];
ry(1.9404116247350602) q[2];
ry(-1.770191204647302) q[3];
cx q[2],q[3];
ry(-0.29974132332230835) q[4];
ry(1.4119099488161015) q[5];
cx q[4],q[5];
ry(-1.4951934425274978) q[4];
ry(-1.4974615091966812) q[5];
cx q[4],q[5];
ry(-1.6155120177679365) q[6];
ry(0.12400183088726151) q[7];
cx q[6],q[7];
ry(0.05102727087912014) q[6];
ry(-1.5777698848909996) q[7];
cx q[6],q[7];
ry(-0.05406772843192087) q[8];
ry(-1.379104120837038) q[9];
cx q[8],q[9];
ry(-0.022460512196484518) q[8];
ry(-1.560450053132854) q[9];
cx q[8],q[9];
ry(-2.0204987556628886) q[10];
ry(-2.0998262383776476) q[11];
cx q[10],q[11];
ry(0.007651704085317131) q[10];
ry(-0.003458709400554483) q[11];
cx q[10],q[11];
ry(3.1010459942050272) q[12];
ry(1.589847452913629) q[13];
cx q[12],q[13];
ry(0.036487326679730495) q[12];
ry(-0.30104688256850576) q[13];
cx q[12],q[13];
ry(-2.1818596342891223) q[14];
ry(1.1022885503178426) q[15];
cx q[14],q[15];
ry(-2.006857101887512) q[14];
ry(-1.1810269541965195) q[15];
cx q[14],q[15];
ry(-0.892630288892188) q[16];
ry(0.625412159846084) q[17];
cx q[16],q[17];
ry(1.9061732112374665) q[16];
ry(0.5487484326114656) q[17];
cx q[16],q[17];
ry(1.014881880239555) q[18];
ry(-0.7225527729167096) q[19];
cx q[18],q[19];
ry(-3.0772852840520897) q[18];
ry(0.11504232180750328) q[19];
cx q[18],q[19];
ry(2.102208409945943) q[0];
ry(0.906845468407699) q[2];
cx q[0],q[2];
ry(-2.875293123965456) q[0];
ry(-2.714343550442808) q[2];
cx q[0],q[2];
ry(-1.5266139359783526) q[2];
ry(1.4705526936767095) q[4];
cx q[2],q[4];
ry(0.08275064433082324) q[2];
ry(-0.09656076447906299) q[4];
cx q[2],q[4];
ry(-2.9817032373720034) q[4];
ry(0.003357190578776148) q[6];
cx q[4],q[6];
ry(-1.5717820838820515) q[4];
ry(-1.5738552363313156) q[6];
cx q[4],q[6];
ry(2.225476504878698) q[6];
ry(1.6459959767788241) q[8];
cx q[6],q[8];
ry(-0.0002782408824560485) q[6];
ry(3.1352577120395235) q[8];
cx q[6],q[8];
ry(3.0516940333617506) q[8];
ry(0.2675886053123393) q[10];
cx q[8],q[10];
ry(-3.139754662584058) q[8];
ry(-1.5806540303134708) q[10];
cx q[8],q[10];
ry(2.6658783349297743) q[10];
ry(0.03473382668819608) q[12];
cx q[10],q[12];
ry(-0.00012091535406622) q[10];
ry(-3.1321714938871703) q[12];
cx q[10],q[12];
ry(-0.973233600990138) q[12];
ry(-2.1650338340358504) q[14];
cx q[12],q[14];
ry(3.120869699246538) q[12];
ry(-3.108714988662774) q[14];
cx q[12],q[14];
ry(-0.2681309631401687) q[14];
ry(0.6834711299751149) q[16];
cx q[14],q[16];
ry(-0.008961474607387741) q[14];
ry(0.0055739974534666226) q[16];
cx q[14],q[16];
ry(-0.9813831526552669) q[16];
ry(2.8163540671576746) q[18];
cx q[16],q[18];
ry(1.5363921856039315) q[16];
ry(1.98625809450665) q[18];
cx q[16],q[18];
ry(2.7536175148189375) q[1];
ry(-2.7249779845910647) q[3];
cx q[1],q[3];
ry(2.7169103591269024) q[1];
ry(-2.5234628250182727) q[3];
cx q[1],q[3];
ry(-1.3918711757079323) q[3];
ry(0.6072115692624714) q[5];
cx q[3],q[5];
ry(3.1376741718896426) q[3];
ry(3.132034004010761) q[5];
cx q[3],q[5];
ry(0.32294190187081545) q[5];
ry(1.418372334807975) q[7];
cx q[5],q[7];
ry(1.5729074202248112) q[5];
ry(-0.0007612761027930848) q[7];
cx q[5],q[7];
ry(-1.5905792326185224) q[7];
ry(0.8313222514657638) q[9];
cx q[7],q[9];
ry(0.00423703482867027) q[7];
ry(-0.24081988731530846) q[9];
cx q[7],q[9];
ry(3.112418557536351) q[9];
ry(0.5189089706504862) q[11];
cx q[9],q[11];
ry(-1.572597188828703) q[9];
ry(0.0023233672922033243) q[11];
cx q[9],q[11];
ry(1.5717288204351538) q[11];
ry(-0.41574972100252694) q[13];
cx q[11],q[13];
ry(-3.1367187807176204) q[11];
ry(2.972099756619316) q[13];
cx q[11],q[13];
ry(0.36812813902506436) q[13];
ry(1.2035926140254891) q[15];
cx q[13],q[15];
ry(2.434593980753123) q[13];
ry(2.2542041666094277) q[15];
cx q[13],q[15];
ry(-1.4649921186774764) q[15];
ry(-1.5536226825011183) q[17];
cx q[15],q[17];
ry(-1.587453823013231) q[15];
ry(-3.140844746583921) q[17];
cx q[15],q[17];
ry(2.5679687727529403) q[17];
ry(-2.772461705146023) q[19];
cx q[17],q[19];
ry(3.1274337599993913) q[17];
ry(0.0012975943075694248) q[19];
cx q[17],q[19];
ry(-2.9763877874187403) q[0];
ry(2.3036949856689892) q[3];
cx q[0],q[3];
ry(-0.21245725444629313) q[0];
ry(-2.6073410420889465) q[3];
cx q[0],q[3];
ry(0.12941407587616105) q[1];
ry(-0.6936196803548486) q[2];
cx q[1],q[2];
ry(1.7612095133114853) q[1];
ry(2.173400960524397) q[2];
cx q[1],q[2];
ry(-1.4586349096289315) q[2];
ry(2.9832271146178266) q[5];
cx q[2],q[5];
ry(0.0028330603787584856) q[2];
ry(-3.1138999407844374) q[5];
cx q[2],q[5];
ry(-2.630910416281082) q[3];
ry(2.0707594163866965) q[4];
cx q[3],q[4];
ry(0.0040749090019591705) q[3];
ry(3.13982705817971) q[4];
cx q[3],q[4];
ry(-0.8318985044299616) q[4];
ry(1.0331754358547856) q[7];
cx q[4],q[7];
ry(-0.0021497544009374536) q[4];
ry(0.06877991641625059) q[7];
cx q[4],q[7];
ry(-1.1432993216972565) q[5];
ry(-1.0050375147326192) q[6];
cx q[5],q[6];
ry(3.120930553529485) q[5];
ry(-2.2001935281549487) q[6];
cx q[5],q[6];
ry(0.05932538021321743) q[6];
ry(-0.6486398582896076) q[9];
cx q[6],q[9];
ry(-0.00030643024568325077) q[6];
ry(-3.1403917387366223) q[9];
cx q[6],q[9];
ry(-1.1052074287013123) q[7];
ry(3.130261765564799) q[8];
cx q[7],q[8];
ry(1.5702125614714717) q[7];
ry(-0.061573297889072544) q[8];
cx q[7],q[8];
ry(2.0089318109089587) q[8];
ry(-0.9020254370645633) q[11];
cx q[8],q[11];
ry(0.001240409803826052) q[8];
ry(0.0030132368588997593) q[11];
cx q[8],q[11];
ry(3.0620657500324215) q[9];
ry(1.2942123232357465) q[10];
cx q[9],q[10];
ry(-1.8417166403471497) q[9];
ry(0.011684931416845679) q[10];
cx q[9],q[10];
ry(-2.8050013386441632) q[10];
ry(-2.7190615218225482) q[13];
cx q[10],q[13];
ry(0.022052470057460492) q[10];
ry(3.140136558866315) q[13];
cx q[10],q[13];
ry(0.9040711447579302) q[11];
ry(2.3423737186790654) q[12];
cx q[11],q[12];
ry(3.140926512907895) q[11];
ry(-1.551833491033941) q[12];
cx q[11],q[12];
ry(-0.7696017001814219) q[12];
ry(3.034283757437685) q[15];
cx q[12],q[15];
ry(3.1383813838752004) q[12];
ry(-3.092360619885894) q[15];
cx q[12],q[15];
ry(1.550601097710764) q[13];
ry(0.991642881619071) q[14];
cx q[13],q[14];
ry(-0.6215595476079052) q[13];
ry(-1.2697719314666562) q[14];
cx q[13],q[14];
ry(-1.753862139160557) q[14];
ry(0.4970816839301547) q[17];
cx q[14],q[17];
ry(3.1341534988102184) q[14];
ry(3.1370016971078227) q[17];
cx q[14],q[17];
ry(-1.775725994341797) q[15];
ry(-2.7844181384174336) q[16];
cx q[15],q[16];
ry(-0.016916779497673673) q[15];
ry(3.1252165284703692) q[16];
cx q[15],q[16];
ry(3.132308375119653) q[16];
ry(-1.031926596494677) q[19];
cx q[16],q[19];
ry(2.101779879761377) q[16];
ry(1.8455199476622048) q[19];
cx q[16],q[19];
ry(1.4857378109647605) q[17];
ry(1.4642803515426408) q[18];
cx q[17],q[18];
ry(-0.010780107888544851) q[17];
ry(-0.7657288113389679) q[18];
cx q[17],q[18];
ry(-3.1108985457049068) q[0];
ry(-1.4308023118446105) q[1];
cx q[0],q[1];
ry(-2.151278450747558) q[0];
ry(2.180170459427656) q[1];
cx q[0],q[1];
ry(-1.1191629419891396) q[2];
ry(-1.1187092858272802) q[3];
cx q[2],q[3];
ry(-1.3872809882551191) q[2];
ry(0.11662450624913671) q[3];
cx q[2],q[3];
ry(-2.808925899236175) q[4];
ry(-0.8050262262131431) q[5];
cx q[4],q[5];
ry(0.0019749405931124286) q[4];
ry(-1.5753312436818954) q[5];
cx q[4],q[5];
ry(-2.9591050152850555) q[6];
ry(-1.3075405715910509) q[7];
cx q[6],q[7];
ry(3.140625575462091) q[6];
ry(-1.5686686979950422) q[7];
cx q[6],q[7];
ry(-0.360154137090956) q[8];
ry(2.862102897788821) q[9];
cx q[8],q[9];
ry(-1.5687077834345038) q[8];
ry(-0.04113832456646715) q[9];
cx q[8],q[9];
ry(-1.2222049956031853) q[10];
ry(-0.004150158553837713) q[11];
cx q[10],q[11];
ry(1.5692537847660397) q[10];
ry(1.568238765344494) q[11];
cx q[10],q[11];
ry(2.1902329800480196) q[12];
ry(1.5275481930357149) q[13];
cx q[12],q[13];
ry(-1.555492772681002) q[12];
ry(1.5799513475286973) q[13];
cx q[12],q[13];
ry(-0.32232734506796135) q[14];
ry(-0.46698581304940046) q[15];
cx q[14],q[15];
ry(1.5672891198667291) q[14];
ry(-2.0715324953377916) q[15];
cx q[14],q[15];
ry(1.0209392041825156) q[16];
ry(-2.0879879271459787) q[17];
cx q[16],q[17];
ry(3.1393024454659204) q[16];
ry(-3.1344271268192823) q[17];
cx q[16],q[17];
ry(-0.9097668244725988) q[18];
ry(0.6561525825028535) q[19];
cx q[18],q[19];
ry(-1.1571780181223348) q[18];
ry(-1.5255037976476724) q[19];
cx q[18],q[19];
ry(-2.7511083395511857) q[0];
ry(-1.4843232084594105) q[2];
cx q[0],q[2];
ry(-1.6926689767105494) q[0];
ry(2.4465766191125993) q[2];
cx q[0],q[2];
ry(-1.3474803248941019) q[2];
ry(-2.2424747640633536) q[4];
cx q[2],q[4];
ry(-0.0028182315798224167) q[2];
ry(-3.1401777923943404) q[4];
cx q[2],q[4];
ry(0.5704971830326054) q[4];
ry(2.785255344692581) q[6];
cx q[4],q[6];
ry(0.004761017537640911) q[4];
ry(-3.1305705520084866) q[6];
cx q[4],q[6];
ry(1.9884099890933333) q[6];
ry(1.9517291856349877) q[8];
cx q[6],q[8];
ry(-0.0007341037434649422) q[6];
ry(3.1395922273879324) q[8];
cx q[6],q[8];
ry(1.7449620339235243) q[8];
ry(-3.1410622941618342) q[10];
cx q[8],q[10];
ry(1.571063195402422) q[8];
ry(3.139767958307995) q[10];
cx q[8],q[10];
ry(1.5295457328571826) q[10];
ry(-1.2558052469665881) q[12];
cx q[10],q[12];
ry(-3.1410776734554844) q[10];
ry(-0.037597449780808034) q[12];
cx q[10],q[12];
ry(2.806945742322256) q[12];
ry(-2.4940910635852855) q[14];
cx q[12],q[14];
ry(-0.022563242725242873) q[12];
ry(-2.4409700337298275) q[14];
cx q[12],q[14];
ry(-1.0134770832513638) q[14];
ry(-0.6981305491031513) q[16];
cx q[14],q[16];
ry(3.0940928114907567) q[14];
ry(-3.1240357076281464) q[16];
cx q[14],q[16];
ry(0.07497466550737375) q[16];
ry(-1.6931779431211167) q[18];
cx q[16],q[18];
ry(-1.4571014048000777) q[16];
ry(1.2356498093587618) q[18];
cx q[16],q[18];
ry(-0.08495223418872168) q[1];
ry(-0.45787732751829235) q[3];
cx q[1],q[3];
ry(1.039150311283981) q[1];
ry(1.498491608368283) q[3];
cx q[1],q[3];
ry(1.3206049867309928) q[3];
ry(-0.7580040754679516) q[5];
cx q[3],q[5];
ry(2.7677197288090833) q[3];
ry(-3.1264076005611177) q[5];
cx q[3],q[5];
ry(-1.3939950904389624) q[5];
ry(-0.2815887222506332) q[7];
cx q[5],q[7];
ry(-0.005116580289579176) q[5];
ry(-3.1398564259846764) q[7];
cx q[5],q[7];
ry(1.3818836362040061) q[7];
ry(-1.571298346077378) q[9];
cx q[7],q[9];
ry(-1.4526647865266078) q[7];
ry(3.138449014519949) q[9];
cx q[7],q[9];
ry(1.5692696375956965) q[9];
ry(0.43901910953225537) q[11];
cx q[9],q[11];
ry(-3.1411836387095184) q[9];
ry(-1.5703935307371628) q[11];
cx q[9],q[11];
ry(3.006112912916022) q[11];
ry(2.5907439896793485) q[13];
cx q[11],q[13];
ry(-0.004529471626690966) q[11];
ry(-0.0003854758325632846) q[13];
cx q[11],q[13];
ry(-2.4751824816526278) q[13];
ry(-1.2209841736845632) q[15];
cx q[13],q[15];
ry(0.005210363209996238) q[13];
ry(0.03447454273939954) q[15];
cx q[13],q[15];
ry(-2.5949662459250717) q[15];
ry(2.5805541265637455) q[17];
cx q[15],q[17];
ry(0.3522015303210511) q[15];
ry(0.013794760752811719) q[17];
cx q[15],q[17];
ry(-2.81595622311009) q[17];
ry(0.020084634423156847) q[19];
cx q[17],q[19];
ry(-0.003681206739593712) q[17];
ry(-0.01394449121548055) q[19];
cx q[17],q[19];
ry(2.136180626732675) q[0];
ry(1.5502100644454628) q[3];
cx q[0],q[3];
ry(1.7814146324146614) q[0];
ry(-2.450777970131849) q[3];
cx q[0],q[3];
ry(-1.7753684609594127) q[1];
ry(-2.430773313107239) q[2];
cx q[1],q[2];
ry(-0.7974160046501366) q[1];
ry(1.55264187737122) q[2];
cx q[1],q[2];
ry(-1.0502855483189528) q[2];
ry(-1.399281427433504) q[5];
cx q[2],q[5];
ry(0.5016178469104506) q[2];
ry(0.030146073593879663) q[5];
cx q[2],q[5];
ry(-2.3107271258920283) q[3];
ry(-0.11496296985193673) q[4];
cx q[3],q[4];
ry(3.137586577168883) q[3];
ry(-3.140630093696262) q[4];
cx q[3],q[4];
ry(-1.5862443110554336) q[4];
ry(2.4927959552240595) q[7];
cx q[4],q[7];
ry(-0.4145220025207967) q[4];
ry(1.6358156598975047) q[7];
cx q[4],q[7];
ry(1.84637849093515) q[5];
ry(0.059603348789322406) q[6];
cx q[5],q[6];
ry(1.6061428129889166) q[5];
ry(0.002109557135031868) q[6];
cx q[5],q[6];
ry(0.04014378737197699) q[6];
ry(2.21241687729364) q[9];
cx q[6],q[9];
ry(0.008385040596600541) q[6];
ry(-0.0035302179139726775) q[9];
cx q[6],q[9];
ry(0.012106425323701764) q[7];
ry(-0.9785016780160228) q[8];
cx q[7],q[8];
ry(-1.567377479080555) q[7];
ry(-0.003846602014263567) q[8];
cx q[7],q[8];
ry(-0.13241267131492318) q[8];
ry(-0.4390660568053671) q[11];
cx q[8],q[11];
ry(8.057877449729745e-07) q[8];
ry(0.0020534395253791102) q[11];
cx q[8],q[11];
ry(0.8458344139308803) q[9];
ry(-1.5305398615018206) q[10];
cx q[9],q[10];
ry(1.562460079747435) q[9];
ry(0.004635438808650526) q[10];
cx q[9],q[10];
ry(-3.1410168522799085) q[10];
ry(-0.10578323820550928) q[13];
cx q[10],q[13];
ry(1.5717870833864491) q[10];
ry(-1.5713911682425026) q[13];
cx q[10],q[13];
ry(1.4401541328868248) q[11];
ry(-2.9673141118266306) q[12];
cx q[11],q[12];
ry(-1.5726304444752852) q[11];
ry(-3.1413567378008826) q[12];
cx q[11],q[12];
ry(0.9005025645311475) q[12];
ry(0.5960193793867808) q[15];
cx q[12],q[15];
ry(3.1292626413864015) q[12];
ry(-3.1391363281392866) q[15];
cx q[12],q[15];
ry(1.7721107509341765) q[13];
ry(1.2659787334384027) q[14];
cx q[13],q[14];
ry(0.0015510832161629295) q[13];
ry(-3.1415799071020762) q[14];
cx q[13],q[14];
ry(1.5071580400935733) q[14];
ry(-0.6171455534612885) q[17];
cx q[14],q[17];
ry(-0.02436094141060818) q[14];
ry(1.57354279314226) q[17];
cx q[14],q[17];
ry(1.2060161092793127) q[15];
ry(-0.19695733268667048) q[16];
cx q[15],q[16];
ry(-0.0019202288077336909) q[15];
ry(-0.008452397561121039) q[16];
cx q[15],q[16];
ry(-1.8122276970292952) q[16];
ry(-0.3947658666458551) q[19];
cx q[16],q[19];
ry(-0.28492756228877436) q[16];
ry(1.3585244265503487) q[19];
cx q[16],q[19];
ry(-2.701503554920567) q[17];
ry(-3.005587344778206) q[18];
cx q[17],q[18];
ry(-3.1401439071861685) q[17];
ry(3.0993593443223815) q[18];
cx q[17],q[18];
ry(3.1078330499900093) q[0];
ry(-0.3748789103542629) q[1];
cx q[0],q[1];
ry(-1.8871237041514553) q[0];
ry(-0.2273228382428877) q[1];
cx q[0],q[1];
ry(-1.1306910407878892) q[2];
ry(0.7912589153392855) q[3];
cx q[2],q[3];
ry(-1.5144606888561523) q[2];
ry(1.5927722849729018) q[3];
cx q[2],q[3];
ry(1.3294048439514028) q[4];
ry(-0.47733685739363646) q[5];
cx q[4],q[5];
ry(-0.009420943505889387) q[4];
ry(3.1226325582267824) q[5];
cx q[4],q[5];
ry(1.5308729867025939) q[6];
ry(0.0007076633999911763) q[7];
cx q[6],q[7];
ry(3.140678165403737) q[6];
ry(1.5712345157077374) q[7];
cx q[6],q[7];
ry(-2.8997673068434016) q[8];
ry(1.1842325682709125) q[9];
cx q[8],q[9];
ry(0.0005563381360538359) q[8];
ry(-3.122981297891181) q[9];
cx q[8],q[9];
ry(-3.1413179747345548) q[10];
ry(-2.730998778316728) q[11];
cx q[10],q[11];
ry(1.5696841394712244) q[10];
ry(1.5670175979752594) q[11];
cx q[10],q[11];
ry(-0.6597874036905234) q[12];
ry(-2.997994701451019) q[13];
cx q[12],q[13];
ry(3.11705669630899) q[12];
ry(-1.5705494182901527) q[13];
cx q[12],q[13];
ry(-0.0025730914782586827) q[14];
ry(0.2879342182156722) q[15];
cx q[14],q[15];
ry(1.569324238565919) q[14];
ry(-1.2143753292441446) q[15];
cx q[14],q[15];
ry(2.6823581392164604) q[16];
ry(-1.4462532726531983) q[17];
cx q[16],q[17];
ry(1.5712345121516562) q[16];
ry(-3.130084661441858) q[17];
cx q[16],q[17];
ry(-2.281233057681317) q[18];
ry(2.121808744035703) q[19];
cx q[18],q[19];
ry(-1.0443292992983002) q[18];
ry(-2.798198350866466) q[19];
cx q[18],q[19];
ry(2.3424345225265593) q[0];
ry(-2.736570517299196) q[2];
cx q[0],q[2];
ry(2.93601410515736) q[0];
ry(-0.01579782800534653) q[2];
cx q[0],q[2];
ry(0.9577618331576443) q[2];
ry(0.024579447001568333) q[4];
cx q[2],q[4];
ry(-0.0054158833738746056) q[2];
ry(-3.1243040582073127) q[4];
cx q[2],q[4];
ry(2.8611613436053727) q[4];
ry(0.0159073928814113) q[6];
cx q[4],q[6];
ry(-1.5645880638661238) q[4];
ry(-1.5783882684742427) q[6];
cx q[4],q[6];
ry(1.543126259602853) q[6];
ry(0.175438280039109) q[8];
cx q[6],q[8];
ry(-3.0425596147091127) q[6];
ry(0.16329085055764178) q[8];
cx q[6],q[8];
ry(0.8727855522766683) q[8];
ry(1.6975278349741574) q[10];
cx q[8],q[10];
ry(3.1414230413946815) q[8];
ry(3.1407195014201763) q[10];
cx q[8],q[10];
ry(-2.528540667059754) q[10];
ry(1.5700762989787345) q[12];
cx q[10],q[12];
ry(-1.5788782827595056) q[10];
ry(3.1206605254736037) q[12];
cx q[10],q[12];
ry(-1.5771833068365824) q[12];
ry(0.015720026430895852) q[14];
cx q[12],q[14];
ry(-0.001465253255758192) q[12];
ry(0.05103561994577141) q[14];
cx q[12],q[14];
ry(0.12447026961295214) q[14];
ry(-2.9210718796366377) q[16];
cx q[14],q[16];
ry(3.1305968348793862) q[14];
ry(2.4586222105280355) q[16];
cx q[14],q[16];
ry(0.9201418432012332) q[16];
ry(-1.739781104862586) q[18];
cx q[16],q[18];
ry(-2.0162464690710102) q[16];
ry(-1.5736143685411628) q[18];
cx q[16],q[18];
ry(-0.07127965619541192) q[1];
ry(-2.8265307657651717) q[3];
cx q[1],q[3];
ry(-3.093337970672069) q[1];
ry(-3.0757202816477376) q[3];
cx q[1],q[3];
ry(-2.766130279160324) q[3];
ry(0.3978868859327798) q[5];
cx q[3],q[5];
ry(0.0009039163652563603) q[3];
ry(3.115204575155641) q[5];
cx q[3],q[5];
ry(1.4211940543206056) q[5];
ry(1.10221212157224) q[7];
cx q[5],q[7];
ry(-0.002982064694285872) q[5];
ry(-3.134223190583034) q[7];
cx q[5],q[7];
ry(1.2117141882027231) q[7];
ry(-2.6473623726095186) q[9];
cx q[7],q[9];
ry(-0.008321411575318294) q[7];
ry(-3.139619845208711) q[9];
cx q[7],q[9];
ry(-1.5966811156019431) q[9];
ry(-1.511759037003517) q[11];
cx q[9],q[11];
ry(-3.121273892912019) q[9];
ry(-1.605456966918104) q[11];
cx q[9],q[11];
ry(-3.058676102904445) q[11];
ry(-2.0227114125740275) q[13];
cx q[11],q[13];
ry(0.0018693564555514186) q[11];
ry(0.00681127769434423) q[13];
cx q[11],q[13];
ry(2.2677122372009726) q[13];
ry(-0.008945818941020842) q[15];
cx q[13],q[15];
ry(0.0002570908231260287) q[13];
ry(3.1269769843837456) q[15];
cx q[13],q[15];
ry(0.15170384784887148) q[15];
ry(0.005364956558661568) q[17];
cx q[15],q[17];
ry(-1.574383637979382) q[15];
ry(-0.0017374970766192229) q[17];
cx q[15],q[17];
ry(1.5086305203448047) q[17];
ry(0.6681380271994051) q[19];
cx q[17],q[19];
ry(-0.00453870670354684) q[17];
ry(2.9873735109513238) q[19];
cx q[17],q[19];
ry(-2.9315123095045528) q[0];
ry(2.179261450381504) q[3];
cx q[0],q[3];
ry(0.03696262032162689) q[0];
ry(-1.63078900855083) q[3];
cx q[0],q[3];
ry(2.8029517073888846) q[1];
ry(-2.6650887702464714) q[2];
cx q[1],q[2];
ry(3.1026010543353757) q[1];
ry(-1.6136854112987757) q[2];
cx q[1],q[2];
ry(2.663329488569655) q[2];
ry(0.5758231380939772) q[5];
cx q[2],q[5];
ry(-0.010310795826536481) q[2];
ry(3.137506493111107) q[5];
cx q[2],q[5];
ry(-2.090205287684391) q[3];
ry(1.3709740808699227) q[4];
cx q[3],q[4];
ry(-0.01982879935190666) q[3];
ry(-2.5179206585966614) q[4];
cx q[3],q[4];
ry(2.082172427511084) q[4];
ry(-1.7030615089177799) q[7];
cx q[4],q[7];
ry(-0.0013254253592416276) q[4];
ry(0.0008975537783582799) q[7];
cx q[4],q[7];
ry(-0.5237825205881717) q[5];
ry(1.6334085727168937) q[6];
cx q[5],q[6];
ry(1.5838598086690383) q[5];
ry(1.5555030251097373) q[6];
cx q[5],q[6];
ry(-0.023757559766544034) q[6];
ry(-2.182961394288898) q[9];
cx q[6],q[9];
ry(3.135274520566487) q[6];
ry(0.014592450852077912) q[9];
cx q[6],q[9];
ry(0.8667588278474302) q[7];
ry(-2.3347216423586166) q[8];
cx q[7],q[8];
ry(1.573444142873579) q[7];
ry(-1.5725388338288355) q[8];
cx q[7],q[8];
ry(1.6107085251764341) q[8];
ry(3.118013219582771) q[11];
cx q[8],q[11];
ry(-1.571275959586579) q[8];
ry(1.5582165379629058) q[11];
cx q[8],q[11];
ry(-2.41909290038341) q[9];
ry(-1.1050013625707396) q[10];
cx q[9],q[10];
ry(1.5673687870017563) q[9];
ry(-0.005141231254548765) q[10];
cx q[9],q[10];
ry(-1.5047232751260022) q[10];
ry(3.105756748522538) q[13];
cx q[10],q[13];
ry(3.141425925536166) q[10];
ry(-3.1373333592245523) q[13];
cx q[10],q[13];
ry(-2.4610882293967755) q[11];
ry(-1.4692584833597442) q[12];
cx q[11],q[12];
ry(3.136066154492656) q[11];
ry(-0.0007783273837356219) q[12];
cx q[11],q[12];
ry(1.2123967623820597) q[12];
ry(1.4332197188983153) q[15];
cx q[12],q[15];
ry(1.5726204476196572) q[12];
ry(-3.137869872836778) q[15];
cx q[12],q[15];
ry(-1.9087209115658226) q[13];
ry(-3.022983248137206) q[14];
cx q[13],q[14];
ry(1.5706735145649722) q[13];
ry(-1.5701774728046907) q[14];
cx q[13],q[14];
ry(0.06634981049066874) q[14];
ry(-1.6343198758494086) q[17];
cx q[14],q[17];
ry(-1.5706541351216017) q[14];
ry(-0.0006000727397916437) q[17];
cx q[14],q[17];
ry(0.0016804888501740223) q[15];
ry(-1.569461315905543) q[16];
cx q[15],q[16];
ry(-1.5711697303321057) q[15];
ry(1.5726129743898163) q[16];
cx q[15],q[16];
ry(-3.134345296167048) q[16];
ry(2.74062891611623) q[19];
cx q[16],q[19];
ry(0.00021655558151056908) q[16];
ry(3.1092956373108005) q[19];
cx q[16],q[19];
ry(-3.1411754618411023) q[17];
ry(1.521332175786883) q[18];
cx q[17],q[18];
ry(-1.5707696050403406) q[17];
ry(-1.5690150276554355) q[18];
cx q[17],q[18];
ry(-1.1969243336541084) q[0];
ry(2.1529795590625453) q[1];
ry(1.16697266428565) q[2];
ry(-1.5605068401261635) q[3];
ry(1.2194595154166694) q[4];
ry(1.5852472666440978) q[5];
ry(0.020186635445588543) q[6];
ry(-1.5713344173118688) q[7];
ry(0.000965299332942614) q[8];
ry(1.8353387060447766) q[9];
ry(1.6357161073265827) q[10];
ry(-0.6615325410499245) q[11];
ry(1.1153024517742693) q[12];
ry(-1.5694937662496817) q[13];
ry(-1.6359418608524303) q[14];
ry(1.5746572675306345) q[15];
ry(0.008434234987038991) q[16];
ry(3.080570272633465) q[17];
ry(3.1405975550395824) q[18];
ry(2.9203147954079975) q[19];