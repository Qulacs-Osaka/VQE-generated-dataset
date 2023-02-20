OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-2.025545634869332) q[0];
rz(-1.5273831660443344) q[0];
ry(0.8068843592876451) q[1];
rz(1.6497876377708094) q[1];
ry(2.869667734293558) q[2];
rz(-0.8632143755413617) q[2];
ry(2.6957629216610375) q[3];
rz(0.27391182626668886) q[3];
ry(2.385432218427844) q[4];
rz(3.0117253899224785) q[4];
ry(-0.2866646648344382) q[5];
rz(-0.7378709140476024) q[5];
ry(0.0030331468808069595) q[6];
rz(-0.6948486404104071) q[6];
ry(-3.1381972489334333) q[7];
rz(-1.8117546938867974) q[7];
ry(-0.7747779508660404) q[8];
rz(-2.851983341391336) q[8];
ry(-2.2251618618240423) q[9];
rz(2.3171178745140733) q[9];
ry(2.4852955150390144) q[10];
rz(0.8200809658181086) q[10];
ry(3.09895521888785) q[11];
rz(1.7709006167692571) q[11];
ry(-0.025847000664037122) q[12];
rz(-1.8992042451731068) q[12];
ry(-0.0069678919116801215) q[13];
rz(-0.5392728670366155) q[13];
ry(0.21266890795459462) q[14];
rz(0.34076274765472275) q[14];
ry(1.275239783590791) q[15];
rz(-0.011450539289501018) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.2607945648065751) q[0];
rz(-2.30486119212256) q[0];
ry(0.042364080054365906) q[1];
rz(-1.8925403432185308) q[1];
ry(-1.3187956718919525) q[2];
rz(1.4645517648871937) q[2];
ry(-1.4554017822718364) q[3];
rz(-2.427272942427265) q[3];
ry(2.080695577426938) q[4];
rz(1.2466383107074532) q[4];
ry(-2.8810201283098213) q[5];
rz(2.3195890783650412) q[5];
ry(-3.1394773207910713) q[6];
rz(0.295230416707707) q[6];
ry(0.0006387892859441918) q[7];
rz(-0.26903521604167857) q[7];
ry(0.021757327177182262) q[8];
rz(-1.967731589243997) q[8];
ry(2.1746131804822078) q[9];
rz(-1.2931486911454444) q[9];
ry(2.5308755622337005) q[10];
rz(-2.107183935327671) q[10];
ry(3.0633829773572967) q[11];
rz(-2.6462830139702147) q[11];
ry(1.5586090494774076) q[12];
rz(-1.5770693567521517) q[12];
ry(1.5672296571713558) q[13];
rz(-1.564939142603687) q[13];
ry(-0.07255364223200421) q[14];
rz(-2.8667836592008165) q[14];
ry(-1.2590215961800872) q[15];
rz(-0.159920494430731) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.3022356632220597) q[0];
rz(-0.32008959091463773) q[0];
ry(-1.653322056259103) q[1];
rz(-2.934762637299862) q[1];
ry(-1.5895954474529201) q[2];
rz(-1.9390810192582977) q[2];
ry(-3.0481296748081546) q[3];
rz(2.235587962506639) q[3];
ry(0.3665457816243956) q[4];
rz(-1.5398852103876077) q[4];
ry(-3.1085412392169984) q[5];
rz(-2.5150904024289376) q[5];
ry(0.012764128694713328) q[6];
rz(2.7866438692769595) q[6];
ry(-3.137729769130086) q[7];
rz(2.307504540205434) q[7];
ry(-1.5288287686869497) q[8];
rz(-1.6468867202043826) q[8];
ry(-2.5299955753307204) q[9];
rz(-2.9908553834745004) q[9];
ry(-3.0506970901080614) q[10];
rz(1.6686626182034543) q[10];
ry(-3.0396473948451024) q[11];
rz(-0.7026976042707025) q[11];
ry(-1.568579935011468) q[12];
rz(2.4486475070706777) q[12];
ry(-1.5720434539734525) q[13];
rz(-2.29567932541698) q[13];
ry(-0.4590814363673017) q[14];
rz(-1.2021072464851046) q[14];
ry(-3.0722895758577993) q[15];
rz(-2.4371662135456855) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.6403909458850406) q[0];
rz(-0.2825663033861048) q[0];
ry(1.3584392294411574) q[1];
rz(2.1804456447552845) q[1];
ry(1.183115725477528) q[2];
rz(3.0426005639198856) q[2];
ry(-3.0553665390781806) q[3];
rz(0.40293668025005847) q[3];
ry(1.3365913864979415) q[4];
rz(-2.5341052803259405) q[4];
ry(1.0489855426668921) q[5];
rz(2.3398221754439334) q[5];
ry(-3.13962566589869) q[6];
rz(-2.5665791249678467) q[6];
ry(0.00491091276078647) q[7];
rz(-1.7989006480839294) q[7];
ry(-2.4415584210225703) q[8];
rz(-2.2573876743502392) q[8];
ry(-0.5295717619707252) q[9];
rz(-2.1439199558270126) q[9];
ry(-0.269931802946308) q[10];
rz(0.2967757272214673) q[10];
ry(-3.137385829139166) q[11];
rz(2.1323646977043524) q[11];
ry(-3.1316850544758736) q[12];
rz(0.1281470839553728) q[12];
ry(0.006436028660775553) q[13];
rz(0.05955101291053477) q[13];
ry(-0.15038003366374927) q[14];
rz(2.151262372621602) q[14];
ry(-2.2781173271212354) q[15];
rz(3.141537434888933) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.034132713087502) q[0];
rz(-3.102002526608426) q[0];
ry(-1.4383714076364038) q[1];
rz(-1.3505775303545784) q[1];
ry(-0.024644044723469267) q[2];
rz(-0.4226962747941787) q[2];
ry(-3.1334864654090477) q[3];
rz(1.865025388754349) q[3];
ry(0.02494229736965803) q[4];
rz(-0.7737439205717178) q[4];
ry(-0.012040434998531151) q[5];
rz(2.7610232441910343) q[5];
ry(1.6266073318808982) q[6];
rz(-1.6282177729643692) q[6];
ry(0.8936945025999226) q[7];
rz(0.5448163244467591) q[7];
ry(1.6368650615015299) q[8];
rz(1.626838449050906) q[8];
ry(-2.1350149970258654) q[9];
rz(2.0110793738644004) q[9];
ry(0.2667356638985803) q[10];
rz(-2.621495217031559) q[10];
ry(-1.4684505909620553) q[11];
rz(-1.7407265476779585) q[11];
ry(-1.568988872516632) q[12];
rz(3.082879291006066) q[12];
ry(1.5829776845788288) q[13];
rz(-3.0358408637014094) q[13];
ry(-3.0794536761188525) q[14];
rz(-0.5355396651174414) q[14];
ry(2.218481851703414) q[15];
rz(-1.758405414129351) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.4645851558010623) q[0];
rz(0.24062325786907746) q[0];
ry(-2.731143358145566) q[1];
rz(-3.119622700321264) q[1];
ry(-0.27833401819941145) q[2];
rz(-2.029622752229686) q[2];
ry(2.454153307518838) q[3];
rz(2.1096862968009296) q[3];
ry(3.0924739347693353) q[4];
rz(-0.5827675851779919) q[4];
ry(2.1478414533279953) q[5];
rz(-1.1888716327118845) q[5];
ry(-0.0034146539146900122) q[6];
rz(1.033098963639584) q[6];
ry(3.140014915113436) q[7];
rz(0.28721186606687354) q[7];
ry(3.1230169044303975) q[8];
rz(-2.114471547964307) q[8];
ry(0.06502056879175043) q[9];
rz(0.7970605230892822) q[9];
ry(-0.02775973819250943) q[10];
rz(-1.026626512582422) q[10];
ry(-2.548955419664016) q[11];
rz(2.9864602165880663) q[11];
ry(-0.05195480363567384) q[12];
rz(0.028560081593892853) q[12];
ry(-3.125577725238902) q[13];
rz(1.8226889606488912) q[13];
ry(-1.6184016268482884) q[14];
rz(0.6967049046523927) q[14];
ry(1.6290358224857682) q[15];
rz(-0.42496225178946034) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.7075238647523259) q[0];
rz(-1.8006593471017955) q[0];
ry(1.6393663183328595) q[1];
rz(-2.1618312371134043) q[1];
ry(-3.1380166910315572) q[2];
rz(-1.4528452668964345) q[2];
ry(3.1330265133832675) q[3];
rz(-2.5370768017144174) q[3];
ry(-0.0016828071330371622) q[4];
rz(-0.6850363108371514) q[4];
ry(-3.1157837084157833) q[5];
rz(-1.2101043168960262) q[5];
ry(-1.527369328478) q[6];
rz(-0.6367020710758564) q[6];
ry(1.4834037393498045) q[7];
rz(2.4284272855427544) q[7];
ry(2.8041401327771447) q[8];
rz(1.8750452384369365) q[8];
ry(0.5893313413162947) q[9];
rz(-2.4368196079214943) q[9];
ry(-0.16847656573985237) q[10];
rz(3.13898489379633) q[10];
ry(-1.1173717448995868) q[11];
rz(0.00015255691395635968) q[11];
ry(-1.563027708793597) q[12];
rz(0.20948941589930675) q[12];
ry(1.185944952844108) q[13];
rz(-2.315360742699315) q[13];
ry(-2.682655076288794) q[14];
rz(2.761326903221215) q[14];
ry(-1.5273251934463454) q[15];
rz(-2.9216328532455096) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.3994647352345586) q[0];
rz(-1.0803135105012271) q[0];
ry(-1.632473561016238) q[1];
rz(0.5103882171491487) q[1];
ry(-1.4983566291435704) q[2];
rz(-1.4220313991095566) q[2];
ry(-1.5609537351304263) q[3];
rz(1.4334687009469642) q[3];
ry(3.0551343336131938) q[4];
rz(-1.6970303630769157) q[4];
ry(2.0905202056294416) q[5];
rz(-0.2013595567257518) q[5];
ry(0.0019484684641319916) q[6];
rz(2.7806495596376113) q[6];
ry(-0.08015819656638357) q[7];
rz(0.012494199712335195) q[7];
ry(-2.665417191621745) q[8];
rz(-2.708134730189445) q[8];
ry(3.0292952822129444) q[9];
rz(-2.4689973609974114) q[9];
ry(0.0016747237843737618) q[10];
rz(-2.303831415647443) q[10];
ry(-3.0718578046760827) q[11];
rz(-3.105137538656983) q[11];
ry(-3.1377645357749344) q[12];
rz(-1.7423023874155872) q[12];
ry(-3.140650820313379) q[13];
rz(2.446971731942053) q[13];
ry(2.4179147409164647) q[14];
rz(1.1494969293622044) q[14];
ry(3.0904258122287818) q[15];
rz(-3.011155500911214) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.4705825281630296) q[0];
rz(0.4815350568442814) q[0];
ry(-2.6459262921262767) q[1];
rz(-0.802408406653461) q[1];
ry(0.09763026141847866) q[2];
rz(3.030264530111816) q[2];
ry(0.0008139353565850075) q[3];
rz(0.5979093198261336) q[3];
ry(3.137910214211937) q[4];
rz(-0.16814307825190952) q[4];
ry(-0.016160733868151315) q[5];
rz(-2.827524845200677) q[5];
ry(3.0979740075801683) q[6];
rz(-0.11095033337075931) q[6];
ry(1.6047755802073507) q[7];
rz(0.8056408498775971) q[7];
ry(-3.1129188655987656) q[8];
rz(1.5361521138462177) q[8];
ry(3.133317440563233) q[9];
rz(1.6940062344692022) q[9];
ry(0.012696732288273616) q[10];
rz(-1.5857656937219637) q[10];
ry(2.820063934872729) q[11];
rz(0.001368362645630228) q[11];
ry(-1.5864520674080809) q[12];
rz(-0.021248327060803367) q[12];
ry(-1.7716064062794539) q[13];
rz(-0.48414707863174833) q[13];
ry(-1.776243079862597) q[14];
rz(-0.5121919946912811) q[14];
ry(0.5844278689000291) q[15];
rz(-2.9949400830405155) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.06610250110911853) q[0];
rz(-1.3011501451091125) q[0];
ry(-0.11213358044595) q[1];
rz(-2.2358564838693478) q[1];
ry(0.24143496603411702) q[2];
rz(-3.0512780141084668) q[2];
ry(3.030591784699547) q[3];
rz(-0.4643057084434013) q[3];
ry(-0.09319245937533542) q[4];
rz(-2.306185460740983) q[4];
ry(-0.751330462991656) q[5];
rz(0.5425876516287378) q[5];
ry(-1.5649146217733403) q[6];
rz(-3.1405669727618) q[6];
ry(-1.5136807285523157) q[7];
rz(3.065825213887768) q[7];
ry(0.41226585353393097) q[8];
rz(-1.2844022812252334) q[8];
ry(3.003977619985022) q[9];
rz(1.4151302803345414) q[9];
ry(0.0010310201628598546) q[10];
rz(2.404121588319512) q[10];
ry(-0.1027755248391653) q[11];
rz(1.5211244585908283) q[11];
ry(1.5691612145884433) q[12];
rz(1.569990989732356) q[12];
ry(3.1411418114147716) q[13];
rz(1.0727160923093042) q[13];
ry(2.004895699814668) q[14];
rz(1.5419813632355952) q[14];
ry(-2.568213013523798) q[15];
rz(-0.5162627691708053) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.8926835893577234) q[0];
rz(2.852668166892646) q[0];
ry(0.9728455477804934) q[1];
rz(-0.5498112259629017) q[1];
ry(1.5840990093386143) q[2];
rz(-1.5689527068427547) q[2];
ry(-3.1404808231772696) q[3];
rz(0.08539708131297404) q[3];
ry(-0.005352755031497124) q[4];
rz(1.9063581253496407) q[4];
ry(-3.1407469252075892) q[5];
rz(-2.570342268164463) q[5];
ry(1.5453478617519298) q[6];
rz(-1.4949947897963733) q[6];
ry(1.5784224972641392) q[7];
rz(2.9552568734631395) q[7];
ry(-3.1387267844750517) q[8];
rz(3.118724686314889) q[8];
ry(0.00532097671184063) q[9];
rz(-0.2838617991194967) q[9];
ry(-3.1408903212951436) q[10];
rz(2.299773135269247) q[10];
ry(3.127876646830539) q[11];
rz(-0.04993881179317519) q[11];
ry(-1.5700544211813678) q[12];
rz(0.02414861263507419) q[12];
ry(-1.571734344445865) q[13];
rz(-1.5697993847776566) q[13];
ry(-1.5716981077286214) q[14];
rz(3.139589349447508) q[14];
ry(-1.571171687396613) q[15];
rz(3.1405175305975743) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(3.1399115998562563) q[0];
rz(0.9187432363651421) q[0];
ry(-0.00851086798876466) q[1];
rz(0.7730382257797509) q[1];
ry(-1.5679972329893677) q[2];
rz(2.495064085360486) q[2];
ry(-0.00046834071388438697) q[3];
rz(0.7987084744148628) q[3];
ry(3.1367202472653726) q[4];
rz(1.9692283146942362) q[4];
ry(1.5768505503638535) q[5];
rz(2.3099377440274433) q[5];
ry(-2.384517680624749) q[6];
rz(0.8703142290717976) q[6];
ry(-2.3352137687150893) q[7];
rz(-2.7758281718822175) q[7];
ry(2.7845499876646267) q[8];
rz(-2.519108481584134) q[8];
ry(-3.0382835101259693) q[9];
rz(0.26679632893289895) q[9];
ry(3.107209692036189) q[10];
rz(-1.9301749440490859) q[10];
ry(-1.6292902289250772) q[11];
rz(2.461325506110804) q[11];
ry(3.1409662840292656) q[12];
rz(-0.19739862536553687) q[12];
ry(1.576566595974028) q[13];
rz(-0.2475248248902799) q[13];
ry(1.5731802824163372) q[14];
rz(1.337019472353887) q[14];
ry(1.573189987362985) q[15];
rz(1.3418628358671252) q[15];