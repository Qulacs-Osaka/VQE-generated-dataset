OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.450061175963731) q[0];
rz(-2.6943378151790967) q[0];
ry(-2.190562256751626) q[1];
rz(0.053669938343272876) q[1];
ry(0.5271779183632841) q[2];
rz(0.40854948137109837) q[2];
ry(2.131336413925673) q[3];
rz(-2.9853664683291283) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3851197460381641) q[0];
rz(2.417916206152829) q[0];
ry(1.0512454844803392) q[1];
rz(2.8315588580246644) q[1];
ry(-3.0672273681892666) q[2];
rz(2.4862264397005536) q[2];
ry(2.44916750225114) q[3];
rz(0.26117826586121384) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2307226569092864) q[0];
rz(2.8088028025630485) q[0];
ry(0.7654079395438211) q[1];
rz(-3.0040876636640017) q[1];
ry(-2.4744424032030743) q[2];
rz(-2.5045402033730375) q[2];
ry(-2.2814197377940317) q[3];
rz(-0.9463536176061577) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.4361303015695284) q[0];
rz(-1.7048214388301715) q[0];
ry(-1.4569656355189156) q[1];
rz(1.397259679973442) q[1];
ry(-0.2378639480600963) q[2];
rz(1.25780158283436) q[2];
ry(2.859062389827166) q[3];
rz(3.0642160830786405) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.6433164648163237) q[0];
rz(0.48519824767836006) q[0];
ry(-1.330530532355969) q[1];
rz(-1.2897202575835083) q[1];
ry(-2.2228661688964433) q[2];
rz(1.2737103207533615) q[2];
ry(2.9987655796437784) q[3];
rz(-0.5302838914926742) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.008565281170287) q[0];
rz(2.1823884669369917) q[0];
ry(-1.1657932937703437) q[1];
rz(-0.3452485787186639) q[1];
ry(2.0099765211475216) q[2];
rz(-2.7485165816845205) q[2];
ry(2.165004037804736) q[3];
rz(-1.1382402876172724) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.06252326396693744) q[0];
rz(-2.1210831867341855) q[0];
ry(-2.0434271014468193) q[1];
rz(-0.7250265787576797) q[1];
ry(1.2425094036305717) q[2];
rz(1.039153419765248) q[2];
ry(-1.6565345748571216) q[3];
rz(0.6065852068441373) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.0335360999875949) q[0];
rz(0.43193038702731007) q[0];
ry(-2.2778251116576396) q[1];
rz(-2.099952859977686) q[1];
ry(0.6737885329475359) q[2];
rz(-2.557174315105079) q[2];
ry(1.330383233357871) q[3];
rz(-1.9503980243214682) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.776262523291851) q[0];
rz(-2.292128299524764) q[0];
ry(2.371183022931063) q[1];
rz(-0.22522105752148655) q[1];
ry(-1.6019019333346156) q[2];
rz(1.6149221870123425) q[2];
ry(-2.4667290411693124) q[3];
rz(1.4988673328276283) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.17797210303247546) q[0];
rz(2.314544143359768) q[0];
ry(-2.6103435618293678) q[1];
rz(1.7275123268146355) q[1];
ry(-1.6564666954193303) q[2];
rz(-0.4925995296496799) q[2];
ry(0.9690784414787741) q[3];
rz(2.761674802253946) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.071464912022043) q[0];
rz(-0.6406480448511767) q[0];
ry(-1.0795442617215416) q[1];
rz(0.4889846914117806) q[1];
ry(-2.4661917698546585) q[2];
rz(2.74512384573947) q[2];
ry(-1.8967257063428784) q[3];
rz(-1.5899734890268604) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.5149204499324238) q[0];
rz(-0.17088179702532494) q[0];
ry(3.1398966557659365) q[1];
rz(-2.0616587065030987) q[1];
ry(1.880038305415281) q[2];
rz(1.3563473736686888) q[2];
ry(1.2495407963322285) q[3];
rz(-1.7909793477484257) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.5646055259140763) q[0];
rz(2.0758049604747364) q[0];
ry(0.17493747230326906) q[1];
rz(0.09520307925988993) q[1];
ry(1.353498808339272) q[2];
rz(1.753928744665991) q[2];
ry(1.967572197372957) q[3];
rz(2.450643082269821) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.658373140668299) q[0];
rz(1.4210820626937322) q[0];
ry(1.8704881383194405) q[1];
rz(1.2601990461376413) q[1];
ry(0.8267131827479633) q[2];
rz(1.0204070973509118) q[2];
ry(2.728295871949772) q[3];
rz(-3.0992392563530924) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.3624570612950615) q[0];
rz(3.054478937224062) q[0];
ry(1.2445801047404705) q[1];
rz(-2.6819151762499955) q[1];
ry(2.32609996812116) q[2];
rz(-2.3938628680639917) q[2];
ry(-0.6332037279930791) q[3];
rz(2.8494883509258186) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.6078053431773136) q[0];
rz(1.677807971264806) q[0];
ry(0.8665270683524939) q[1];
rz(-2.42056221518843) q[1];
ry(0.8808084270702334) q[2];
rz(-2.2558880519184408) q[2];
ry(1.8710976067617027) q[3];
rz(0.7895244442837284) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.5345450736570134) q[0];
rz(2.279900867227509) q[0];
ry(-1.8501231767944954) q[1];
rz(3.009995200761934) q[1];
ry(0.3309227754095486) q[2];
rz(2.6797514090441172) q[2];
ry(-0.9325599893954617) q[3];
rz(-1.9651074634845793) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.2179563836937697) q[0];
rz(0.1909933008724092) q[0];
ry(-2.577020548747297) q[1];
rz(-0.6686076226083699) q[1];
ry(0.33232348659451383) q[2];
rz(0.4606392206089964) q[2];
ry(0.7692996937078549) q[3];
rz(1.1948220700844434) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.8606631813254673) q[0];
rz(2.4998912021969746) q[0];
ry(-2.9540656609687326) q[1];
rz(1.2684228745435817) q[1];
ry(2.3271905024264234) q[2];
rz(1.6403027854114338) q[2];
ry(-0.628336396996688) q[3];
rz(1.1680040942694132) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.7101185965153585) q[0];
rz(0.6445171898287025) q[0];
ry(-1.2392337573334578) q[1];
rz(2.791016429948458) q[1];
ry(-2.7493234462941762) q[2];
rz(-1.9998298069304943) q[2];
ry(0.623210472509659) q[3];
rz(-2.5521217798838562) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.23882371991825857) q[0];
rz(-2.3845686342614325) q[0];
ry(-1.4991560495370313) q[1];
rz(-2.3347398695597836) q[1];
ry(-2.003468900733706) q[2];
rz(-1.849450833759981) q[2];
ry(2.6818156304053757) q[3];
rz(0.25329232163613913) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.4329376389630308) q[0];
rz(-0.36737954848813587) q[0];
ry(0.7785025191527364) q[1];
rz(-1.914491818907627) q[1];
ry(2.2506606795631043) q[2];
rz(-1.8029059714425202) q[2];
ry(1.7333417206173083) q[3];
rz(1.9683883503920043) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.0769591009516315) q[0];
rz(1.0588845842152184) q[0];
ry(-2.1044935670869203) q[1];
rz(-1.6882769416383765) q[1];
ry(2.8831236682768875) q[2];
rz(2.2447146443809185) q[2];
ry(0.45674691945957463) q[3];
rz(3.0729679911709464) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.9167380065892377) q[0];
rz(-2.270252060997252) q[0];
ry(2.782143649036058) q[1];
rz(1.6247942732323495) q[1];
ry(0.4501709708746706) q[2];
rz(2.0218246152126134) q[2];
ry(2.7878248130354955) q[3];
rz(-0.7937614746694699) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.1131021755769046) q[0];
rz(-0.8505754654096771) q[0];
ry(-1.2676760687780764) q[1];
rz(1.4647685594325957) q[1];
ry(2.894520113218329) q[2];
rz(-1.169593978579623) q[2];
ry(0.13755724393070423) q[3];
rz(-0.23746684220637457) q[3];