OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.019909012076722427) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.12535444054119757) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09743943286271146) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.10708528449485108) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.19116742367691686) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.024114060195893742) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06310527802286169) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1473359713357652) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.028763032347335585) q[3];
cx q[2],q[3];
rz(-0.010444187801100263) q[0];
rz(-0.005902482576503828) q[1];
rz(-0.09708657399250588) q[2];
rz(-0.09522999842868932) q[3];
rx(-0.6182946187766658) q[0];
rx(-0.189337247931059) q[1];
rx(-0.43291931072020884) q[2];
rx(-0.4864632259485838) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.07308880268073004) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.23844485223751563) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.018923144756857555) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.049537669338417495) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.24949547771437403) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.009508431036392555) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.05472972999984908) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.11941360770942178) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.009011180723307775) q[3];
cx q[2],q[3];
rz(-0.010670946233619997) q[0];
rz(0.00817968252982902) q[1];
rz(-0.01485868351495799) q[2];
rz(-0.04304030641169965) q[3];
rx(-0.6127611815815653) q[0];
rx(-0.19795068855102707) q[1];
rx(-0.45119375869155104) q[2];
rx(-0.5199399034695246) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.005312046305847443) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.30504490775013554) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.10418703807577889) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.07766722098890023) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.39885400673471594) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.16864053938603124) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.07619689599576579) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.25137421665722925) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.037798489927200724) q[3];
cx q[2],q[3];
rz(-0.011576759267342605) q[0];
rz(0.008892370022849047) q[1];
rz(-0.03312079438449902) q[2];
rz(-0.04899158493178091) q[3];
rx(-0.6020368021630162) q[0];
rx(-0.14567924665909354) q[1];
rx(-0.3556364472289851) q[2];
rx(-0.5123918227655339) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05941140322996286) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.37967186997116864) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.07776107772016656) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.09162321015792022) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4343116419618537) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2054248932808888) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.09130740794551616) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1950384957741368) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06463914683933955) q[3];
cx q[2],q[3];
rz(-0.041936441261777736) q[0];
rz(-0.0761798234915493) q[1];
rz(0.018582085492369792) q[2];
rz(0.045212406418955756) q[3];
rx(-0.5483079702280963) q[0];
rx(-0.2152311988897872) q[1];
rx(-0.32323782340055146) q[2];
rx(-0.5129325737307232) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.1043572652726819) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.2960337082640759) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.042298943005353404) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.06928256086141768) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.507110539117335) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.31138822256517734) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.04762187417164038) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.1265999015280061) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02964124556966037) q[3];
cx q[2],q[3];
rz(-0.07274757844951539) q[0];
rz(0.004661097030504024) q[1];
rz(0.13530102099185864) q[2];
rz(0.07947035176845497) q[3];
rx(-0.5826294670835352) q[0];
rx(-0.23307616368818046) q[1];
rx(-0.32808347730471377) q[2];
rx(-0.4981654428729667) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.02983073268873034) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.1849576786484835) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09400577650004684) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.0003526934301176869) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.519062049892453) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3547646551844161) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.0014778982716591167) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.003673782404841117) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.21955872092102866) q[3];
cx q[2],q[3];
rz(-0.07872913132453024) q[0];
rz(0.09690827161160959) q[1];
rz(0.26551458978512626) q[2];
rz(0.12374158901186481) q[3];
rx(-0.6104351020272509) q[0];
rx(-0.212336356442909) q[1];
rx(-0.28919181594209703) q[2];
rx(-0.5348782676749428) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.04192587090887194) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.17062816138903342) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.19882776477431033) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09561914658448939) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.4979250943193371) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.3737444977860212) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.044684880797044524) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.005561961971271704) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.39312031715912066) q[3];
cx q[2],q[3];
rz(-0.14845685798542269) q[0];
rz(0.21301547700187934) q[1];
rz(0.3211737222892964) q[2];
rz(0.08294663216906467) q[3];
rx(-0.6065687940462929) q[0];
rx(-0.2220325112231463) q[1];
rx(-0.3072032848011937) q[2];
rx(-0.5242264739381712) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.027438540614320477) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11656183413511254) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.04237469405059464) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.13523487211572607) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.47806005910058774) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.2492828184875972) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.09910962297385076) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.028389625226063017) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.24217448204503636) q[3];
cx q[2],q[3];
rz(-0.11999050828827178) q[0];
rz(0.2802828879484301) q[1];
rz(0.25252723332933014) q[2];
rz(0.076854940116332) q[3];
rx(-0.5951384407144643) q[0];
rx(-0.26153161116863266) q[1];
rx(-0.3745867350127049) q[2];
rx(-0.6329074131524882) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0938589892964339) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.11187704059021986) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.1134155075885069) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.10635301391997806) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3745749865747058) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13347284312960622) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.047879299167797075) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.0166266036421288) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10420555107717613) q[3];
cx q[2],q[3];
rz(-0.13883279356733436) q[0];
rz(0.27582688992771376) q[1];
rz(0.13205312807299635) q[2];
rz(-0.09597827321853708) q[3];
rx(-0.6403036821778271) q[0];
rx(-0.3069945208458418) q[1];
rx(-0.372518279485459) q[2];
rx(-0.6696395433640455) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.0769679812309521) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.008655671103133323) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.007237705526615272) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.09852029801211587) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.3611114835906633) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.048978406857894875) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.04854899512804386) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.09792342278937617) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.011211471269108075) q[3];
cx q[2],q[3];
rz(-0.16986423058001918) q[0];
rz(0.24466588180015508) q[1];
rz(-0.023374147598543414) q[2];
rz(-0.1252988352347416) q[3];
rx(-0.5395850644338576) q[0];
rx(-0.4159804945563124) q[1];
rx(-0.2794791533619054) q[2];
rx(-0.5647277483100185) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.04401461482279907) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.18199485886315386) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.2234832168222576) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.203348319781424) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.18545687216217555) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.09500239975009696) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.06194204599538133) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.21662791524072283) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1371265616049506) q[3];
cx q[2],q[3];
rz(-0.07547967772687912) q[0];
rz(0.19545425341285047) q[1];
rz(-0.03908715293988263) q[2];
rz(-0.1694490054402703) q[3];
rx(-0.523279379884777) q[0];
rx(-0.5325708892605966) q[1];
rx(-0.2540456140913182) q[2];
rx(-0.5920909868365465) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.047929647389448056) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.16445848477581612) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.19016657234590767) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.2976399024872056) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.05535374248150043) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.028085649295732568) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.04800212891614888) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.20467470049425138) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0843665426295962) q[3];
cx q[2],q[3];
rz(-0.05251805162778694) q[0];
rz(0.25059873790212694) q[1];
rz(-0.11045682706109512) q[2];
rz(-0.2565938178719789) q[3];
rx(-0.501354620035572) q[0];
rx(-0.6077577074861622) q[1];
rx(-0.1944492263531704) q[2];
rx(-0.6145363924221761) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(0.00903838659352123) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.1008810710873331) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.09559187706003007) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.34293832317382156) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.032470808268716025) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.047437666271279245) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.02932460079012159) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.12745256553029216) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10671349058674266) q[3];
cx q[2],q[3];
rz(-0.0808665941765233) q[0];
rz(0.2277729171120555) q[1];
rz(-0.060566360800229584) q[2];
rz(-0.21308451014197874) q[3];
rx(-0.4765776398904897) q[0];
rx(-0.577768804445565) q[1];
rx(-0.17166589419491138) q[2];
rx(-0.556246506848585) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.0015437739577271702) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05054666272747311) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.045945341070959385) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.38526982200103577) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.0015888929907922993) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.006331291634667985) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(0.009207199299052195) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.03192840322547114) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.10754109117943035) q[3];
cx q[2],q[3];
rz(-0.06361294584593115) q[0];
rz(0.19063450297774617) q[1];
rz(-0.09939613498099632) q[2];
rz(-0.17674074195392125) q[3];
rx(-0.4350545035606806) q[0];
rx(-0.6551438582043322) q[1];
rx(-0.15096074005799937) q[2];
rx(-0.552271578364605) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.03341419307127499) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(0.05619899456138717) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.06724150435746297) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.45216805807780924) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.20932299448817024) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.03320863679270657) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.053812961068311224) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.057010732205349614) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11526973937633882) q[3];
cx q[2],q[3];
rz(-0.06913578517137492) q[0];
rz(0.20708647327778673) q[1];
rz(0.006715475863150486) q[2];
rz(-0.1375568606711638) q[3];
rx(-0.49527011155767053) q[0];
rx(-0.6202713456879969) q[1];
rx(-0.16388405036615664) q[2];
rx(-0.5714864753815628) q[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.05948214323413949) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-0.09427716632294987) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.020153743847604925) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.44846144296991625) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.2533050343909621) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.062184236065386304) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.041070243657909206) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.06184023953869085) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.27397841622489294) q[3];
cx q[2],q[3];
rz(-0.18146878181149959) q[0];
rz(0.11694097186560018) q[1];
rz(0.13650159146893046) q[2];
rz(-0.06792278603729718) q[3];
rx(-0.42349305054747005) q[0];
rx(-0.6189580662323122) q[1];
rx(-0.21774844149767786) q[2];
rx(-0.5666971244234154) q[3];