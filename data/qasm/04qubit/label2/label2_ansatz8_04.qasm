OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.04494198125347992) q[0];
ry(0.6057340902014188) q[1];
cx q[0],q[1];
ry(-1.4944146830199694) q[0];
ry(1.4931812060599592) q[1];
cx q[0],q[1];
ry(2.4113957621600863) q[2];
ry(0.0378741967249153) q[3];
cx q[2],q[3];
ry(1.968659658816553) q[2];
ry(-3.044848296656658) q[3];
cx q[2],q[3];
ry(0.6556562228437999) q[0];
ry(-0.5581647973995416) q[2];
cx q[0],q[2];
ry(0.37757271801137776) q[0];
ry(-2.8151260251209895) q[2];
cx q[0],q[2];
ry(-3.0395037476473203) q[1];
ry(-0.028009117796096945) q[3];
cx q[1],q[3];
ry(-2.183459894724691) q[1];
ry(-1.5751046365431922) q[3];
cx q[1],q[3];
ry(-1.8098246238371978) q[0];
ry(1.5072078233567572) q[1];
cx q[0],q[1];
ry(3.0578948448889864) q[0];
ry(-2.850164168197396) q[1];
cx q[0],q[1];
ry(1.4319957866485427) q[2];
ry(-2.3107615853325836) q[3];
cx q[2],q[3];
ry(-2.8846931192992287) q[2];
ry(0.790377608442836) q[3];
cx q[2],q[3];
ry(2.5622038676856542) q[0];
ry(1.8756494579526086) q[2];
cx q[0],q[2];
ry(-2.3046035050205784) q[0];
ry(-0.6503645550150101) q[2];
cx q[0],q[2];
ry(0.13913112048492798) q[1];
ry(-1.906877983511559) q[3];
cx q[1],q[3];
ry(-2.3804604402660634) q[1];
ry(0.8055145280538589) q[3];
cx q[1],q[3];
ry(-0.9597728486942034) q[0];
ry(2.799054449730447) q[1];
cx q[0],q[1];
ry(1.851506404043237) q[0];
ry(-2.601478443217905) q[1];
cx q[0],q[1];
ry(1.3561521401665528) q[2];
ry(-1.659276152789165) q[3];
cx q[2],q[3];
ry(1.075964998342153) q[2];
ry(0.2508294405461222) q[3];
cx q[2],q[3];
ry(1.6168837388048685) q[0];
ry(-2.7270704023186343) q[2];
cx q[0],q[2];
ry(1.0531761459231097) q[0];
ry(1.9641366954352382) q[2];
cx q[0],q[2];
ry(-0.8382131898439598) q[1];
ry(-2.5644696379183833) q[3];
cx q[1],q[3];
ry(-3.101647464411562) q[1];
ry(-1.650808382614067) q[3];
cx q[1],q[3];
ry(-0.14831794334812898) q[0];
ry(0.5083120831762304) q[1];
cx q[0],q[1];
ry(1.3898139469913042) q[0];
ry(-2.3664001230320255) q[1];
cx q[0],q[1];
ry(0.20886381992661462) q[2];
ry(-2.3925755668912774) q[3];
cx q[2],q[3];
ry(-1.9166810194538586) q[2];
ry(1.2632317935764792) q[3];
cx q[2],q[3];
ry(-2.7052290704784023) q[0];
ry(-1.6953188572352937) q[2];
cx q[0],q[2];
ry(-0.5116911029097952) q[0];
ry(-1.8075284477553908) q[2];
cx q[0],q[2];
ry(1.8957685960282884) q[1];
ry(0.2121005314215424) q[3];
cx q[1],q[3];
ry(-2.3862723521172406) q[1];
ry(-1.4694630424232313) q[3];
cx q[1],q[3];
ry(-0.11543340163052258) q[0];
ry(-0.3389638706509053) q[1];
cx q[0],q[1];
ry(-1.789356401205695) q[0];
ry(-2.634298968040143) q[1];
cx q[0],q[1];
ry(-2.8056978305750113) q[2];
ry(1.8870044997294366) q[3];
cx q[2],q[3];
ry(2.621864493471719) q[2];
ry(2.6989098174867214) q[3];
cx q[2],q[3];
ry(-2.6098215983013926) q[0];
ry(2.57412336828707) q[2];
cx q[0],q[2];
ry(2.2638899535389254) q[0];
ry(-2.9189272656955794) q[2];
cx q[0],q[2];
ry(2.788074476139292) q[1];
ry(2.1095068912945507) q[3];
cx q[1],q[3];
ry(2.784667845946438) q[1];
ry(0.05403630295837658) q[3];
cx q[1],q[3];
ry(-1.49041793477539) q[0];
ry(-1.6581836468050462) q[1];
cx q[0],q[1];
ry(-0.3156450623874436) q[0];
ry(-1.5790980277351303) q[1];
cx q[0],q[1];
ry(-2.7794259170419435) q[2];
ry(-2.4470365852968508) q[3];
cx q[2],q[3];
ry(2.82285764464507) q[2];
ry(-0.9128433395580977) q[3];
cx q[2],q[3];
ry(0.3024015043523791) q[0];
ry(0.4963563865908309) q[2];
cx q[0],q[2];
ry(2.2311994693146895) q[0];
ry(2.5796515603498484) q[2];
cx q[0],q[2];
ry(-2.390171898597382) q[1];
ry(-2.1001453760193556) q[3];
cx q[1],q[3];
ry(-1.2379368884892432) q[1];
ry(-1.25828301828257) q[3];
cx q[1],q[3];
ry(-1.305821819226681) q[0];
ry(-2.3329892122127442) q[1];
cx q[0],q[1];
ry(0.7157879951193844) q[0];
ry(-2.2928414443560032) q[1];
cx q[0],q[1];
ry(1.694858989495355) q[2];
ry(1.201937650504167) q[3];
cx q[2],q[3];
ry(0.0021796124128750072) q[2];
ry(2.4354262190470473) q[3];
cx q[2],q[3];
ry(2.1583930644711034) q[0];
ry(2.638659284298363) q[2];
cx q[0],q[2];
ry(2.9105636685330016) q[0];
ry(-1.076034224526789) q[2];
cx q[0],q[2];
ry(-2.602741975639756) q[1];
ry(1.7352366059495228) q[3];
cx q[1],q[3];
ry(2.6062931031747643) q[1];
ry(1.982680436026202) q[3];
cx q[1],q[3];
ry(1.0930432120182898) q[0];
ry(1.9895382234393233) q[1];
ry(2.29729632909835) q[2];
ry(0.09915653342146147) q[3];