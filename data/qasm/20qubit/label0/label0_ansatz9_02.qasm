OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-3.0674811043418706) q[0];
ry(-2.4170023485124075) q[1];
cx q[0],q[1];
ry(-1.187560323092737) q[0];
ry(2.314975961358358) q[1];
cx q[0],q[1];
ry(-2.685846551917141) q[2];
ry(0.4859663677204038) q[3];
cx q[2],q[3];
ry(0.05825483533176179) q[2];
ry(3.0380728980570577) q[3];
cx q[2],q[3];
ry(-2.263671264288816) q[4];
ry(-0.20486055295250072) q[5];
cx q[4],q[5];
ry(2.396384021641725) q[4];
ry(-2.435945476042072) q[5];
cx q[4],q[5];
ry(-1.5451818690582924) q[6];
ry(1.8219849644627786) q[7];
cx q[6],q[7];
ry(-1.9275834592843335) q[6];
ry(-2.5213323134843812) q[7];
cx q[6],q[7];
ry(2.6309521459300518) q[8];
ry(-2.9068907178886856) q[9];
cx q[8],q[9];
ry(-2.9612195335325926) q[8];
ry(2.384987533585989) q[9];
cx q[8],q[9];
ry(-0.8753853515212768) q[10];
ry(-1.0691056951282893) q[11];
cx q[10],q[11];
ry(-1.965979531397485) q[10];
ry(1.1402212084962018) q[11];
cx q[10],q[11];
ry(-1.4001059188160347) q[12];
ry(1.726515651736304) q[13];
cx q[12],q[13];
ry(0.5951320834404954) q[12];
ry(0.04550388127325038) q[13];
cx q[12],q[13];
ry(-0.33204727949756574) q[14];
ry(-0.6444978511485608) q[15];
cx q[14],q[15];
ry(0.5357092619922981) q[14];
ry(2.5549660105535557) q[15];
cx q[14],q[15];
ry(-2.997328930599445) q[16];
ry(-2.369948951271464) q[17];
cx q[16],q[17];
ry(0.06046130048129064) q[16];
ry(0.40180154325999956) q[17];
cx q[16],q[17];
ry(-3.025466111791218) q[18];
ry(0.8294746604105274) q[19];
cx q[18],q[19];
ry(1.7870033918256034) q[18];
ry(0.42802104563646876) q[19];
cx q[18],q[19];
ry(-0.2864878998242988) q[0];
ry(-0.8742268942824023) q[2];
cx q[0],q[2];
ry(1.679004353516087) q[0];
ry(-2.1762704083088122) q[2];
cx q[0],q[2];
ry(1.712830685569266) q[2];
ry(0.6385952949289675) q[4];
cx q[2],q[4];
ry(-2.826464855357976) q[2];
ry(-0.304047306371901) q[4];
cx q[2],q[4];
ry(0.6301366539120677) q[4];
ry(-2.935172281318744) q[6];
cx q[4],q[6];
ry(3.1390026073679054) q[4];
ry(0.0019727216346452792) q[6];
cx q[4],q[6];
ry(-1.3937875283435228) q[6];
ry(1.81318803481321) q[8];
cx q[6],q[8];
ry(4.2794928020686546e-05) q[6];
ry(3.1415918948561963) q[8];
cx q[6],q[8];
ry(-0.7767254443832501) q[8];
ry(-0.17513519811415634) q[10];
cx q[8],q[10];
ry(-1.4137780202747356) q[8];
ry(-1.9796641134059145) q[10];
cx q[8],q[10];
ry(-0.2898878585702205) q[10];
ry(2.665545414169777) q[12];
cx q[10],q[12];
ry(-1.5184041058562068) q[10];
ry(0.2681458583505387) q[12];
cx q[10],q[12];
ry(-1.3240860126899863) q[12];
ry(-2.281775623566764) q[14];
cx q[12],q[14];
ry(-3.0706186466815284) q[12];
ry(0.4027650559175218) q[14];
cx q[12],q[14];
ry(2.7265877233426514) q[14];
ry(-1.8535081485644929) q[16];
cx q[14],q[16];
ry(3.1411827845544953) q[14];
ry(-9.945884058445364e-05) q[16];
cx q[14],q[16];
ry(0.3151541209595398) q[16];
ry(1.425352641657137) q[18];
cx q[16],q[18];
ry(-3.044217623766676) q[16];
ry(-2.844099011449451) q[18];
cx q[16],q[18];
ry(3.1115609826798534) q[1];
ry(2.990350805146144) q[3];
cx q[1],q[3];
ry(-0.1494344599075026) q[1];
ry(0.19544562666773146) q[3];
cx q[1],q[3];
ry(1.0512391448255576) q[3];
ry(1.981472201119507) q[5];
cx q[3],q[5];
ry(-1.5310230673950231) q[3];
ry(-1.509972445277465) q[5];
cx q[3],q[5];
ry(0.12333735999385313) q[5];
ry(0.6684584234884771) q[7];
cx q[5],q[7];
ry(-1.9155394097144076) q[5];
ry(-3.141100863660374) q[7];
cx q[5],q[7];
ry(2.886554000218077) q[7];
ry(2.779254458409953) q[9];
cx q[7],q[9];
ry(-3.141415706460524) q[7];
ry(-0.0004851633579635299) q[9];
cx q[7],q[9];
ry(-2.8135296579607454) q[9];
ry(-2.912667345501235) q[11];
cx q[9],q[11];
ry(-2.9216158804426677) q[9];
ry(2.623140743900142) q[11];
cx q[9],q[11];
ry(-1.9591696774997782) q[11];
ry(-1.83548930607923) q[13];
cx q[11],q[13];
ry(3.1410152596415926) q[11];
ry(3.1204576931558146) q[13];
cx q[11],q[13];
ry(-1.9655817171082788) q[13];
ry(-1.9977682647563355) q[15];
cx q[13],q[15];
ry(-2.9292469428896433) q[13];
ry(-0.26076751228486883) q[15];
cx q[13],q[15];
ry(-2.3924110724654533) q[15];
ry(-1.9441793926292101) q[17];
cx q[15],q[17];
ry(-0.4519679237103009) q[15];
ry(-0.0007074975840613006) q[17];
cx q[15],q[17];
ry(1.8441175240949137) q[17];
ry(-1.297513631047126) q[19];
cx q[17],q[19];
ry(-0.8517848782851392) q[17];
ry(2.609122923067116) q[19];
cx q[17],q[19];
ry(2.5494170944064303) q[0];
ry(-1.3681757957552376) q[3];
cx q[0],q[3];
ry(0.16658167073809893) q[0];
ry(0.12391295687889416) q[3];
cx q[0],q[3];
ry(-1.3834343328478582) q[1];
ry(1.9550972907281006) q[2];
cx q[1],q[2];
ry(3.0167672106326866) q[1];
ry(-0.41884306757348444) q[2];
cx q[1],q[2];
ry(1.5060690441824232) q[2];
ry(-0.17690703540660646) q[5];
cx q[2],q[5];
ry(3.0720022870175714) q[2];
ry(0.019802144124380344) q[5];
cx q[2],q[5];
ry(-0.31891827337389067) q[3];
ry(1.3825567270379517) q[4];
cx q[3],q[4];
ry(-2.6326920603317925) q[3];
ry(0.7172265578325607) q[4];
cx q[3],q[4];
ry(3.1223669480737652) q[4];
ry(2.789835445981205) q[7];
cx q[4],q[7];
ry(0.013959081474322232) q[4];
ry(-3.1318355807304723) q[7];
cx q[4],q[7];
ry(-1.6887294645089623) q[5];
ry(-0.43884699336542177) q[6];
cx q[5],q[6];
ry(-2.2087033364906175) q[5];
ry(0.0018934892855577703) q[6];
cx q[5],q[6];
ry(1.5503712058269599) q[6];
ry(-0.8354223254852954) q[9];
cx q[6],q[9];
ry(-3.140102814938223) q[6];
ry(3.0170455097134554) q[9];
cx q[6],q[9];
ry(-0.9567689815118946) q[7];
ry(0.3743608988958087) q[8];
cx q[7],q[8];
ry(-5.776943070934237e-05) q[7];
ry(2.750508378917559) q[8];
cx q[7],q[8];
ry(0.24505046016431778) q[8];
ry(-1.5827710382016447) q[11];
cx q[8],q[11];
ry(1.1071348787201947) q[8];
ry(-3.1367556862952677) q[11];
cx q[8],q[11];
ry(1.9031538117799847) q[9];
ry(1.083752311740522) q[10];
cx q[9],q[10];
ry(3.139672997999701) q[9];
ry(0.0016968082849478975) q[10];
cx q[9],q[10];
ry(-0.3746819738738676) q[10];
ry(0.4592349643820359) q[13];
cx q[10],q[13];
ry(-0.275234498061141) q[10];
ry(0.6438178302920035) q[13];
cx q[10],q[13];
ry(2.383829354712209) q[11];
ry(-1.6322122139243098) q[12];
cx q[11],q[12];
ry(0.7076492555955147) q[11];
ry(0.046318165744188455) q[12];
cx q[11],q[12];
ry(0.8754572410153276) q[12];
ry(2.578198920695103) q[15];
cx q[12],q[15];
ry(2.6714952492011097) q[12];
ry(0.5555940878428478) q[15];
cx q[12],q[15];
ry(-2.356544295645727) q[13];
ry(2.3932299227385703) q[14];
cx q[13],q[14];
ry(-3.1154014512376706) q[13];
ry(0.9827592762372666) q[14];
cx q[13],q[14];
ry(3.081442004332292) q[14];
ry(0.9692059906057963) q[17];
cx q[14],q[17];
ry(3.141493999435136) q[14];
ry(9.833419843754101e-05) q[17];
cx q[14],q[17];
ry(1.4425270193211392) q[15];
ry(1.0969679483984915) q[16];
cx q[15],q[16];
ry(3.141550633554136) q[15];
ry(-4.952983013950529e-06) q[16];
cx q[15],q[16];
ry(1.3664058312899956) q[16];
ry(-1.2904509282053782) q[19];
cx q[16],q[19];
ry(0.19106213278360837) q[16];
ry(-1.505073352163051) q[19];
cx q[16],q[19];
ry(-0.7418987376232202) q[17];
ry(-0.5472599404651843) q[18];
cx q[17],q[18];
ry(-1.0396895061766476) q[17];
ry(0.6356023891461771) q[18];
cx q[17],q[18];
ry(-2.1803988775829897) q[0];
ry(0.2606271713318918) q[1];
cx q[0],q[1];
ry(2.680449437501569) q[0];
ry(1.9439519667313179) q[1];
cx q[0],q[1];
ry(-0.06889840398961411) q[2];
ry(2.3400172269986452) q[3];
cx q[2],q[3];
ry(-1.2563095999451575) q[2];
ry(1.7262704197927974) q[3];
cx q[2],q[3];
ry(2.8542611390735093) q[4];
ry(-0.8028592232230319) q[5];
cx q[4],q[5];
ry(-3.038203581341842) q[4];
ry(-0.07414623962412124) q[5];
cx q[4],q[5];
ry(1.6130273905211583) q[6];
ry(-0.01844508417574442) q[7];
cx q[6],q[7];
ry(1.5192781113363842) q[6];
ry(1.6234071885380512) q[7];
cx q[6],q[7];
ry(-2.9869068340231464) q[8];
ry(1.220081831704884) q[9];
cx q[8],q[9];
ry(2.640580242975269) q[8];
ry(-2.219819355073257) q[9];
cx q[8],q[9];
ry(0.5602254848585042) q[10];
ry(2.755100769467104) q[11];
cx q[10],q[11];
ry(-2.33224323238193) q[10];
ry(2.812638850338231) q[11];
cx q[10],q[11];
ry(-2.241310911306625) q[12];
ry(-1.1169484212842322) q[13];
cx q[12],q[13];
ry(2.860962129210881) q[12];
ry(1.2801983104249262) q[13];
cx q[12],q[13];
ry(-1.8604128698695035) q[14];
ry(0.8718595994387552) q[15];
cx q[14],q[15];
ry(-1.5732461712157915) q[14];
ry(1.5750906367700503) q[15];
cx q[14],q[15];
ry(-0.3503065864843613) q[16];
ry(2.243038168213952) q[17];
cx q[16],q[17];
ry(-1.1895037665407733) q[16];
ry(-0.6822394573860447) q[17];
cx q[16],q[17];
ry(-1.5876213838436861) q[18];
ry(3.031514490746616) q[19];
cx q[18],q[19];
ry(0.12339943649744976) q[18];
ry(-0.4922681862360632) q[19];
cx q[18],q[19];
ry(-1.0199279255590596) q[0];
ry(1.7345977110694422) q[2];
cx q[0],q[2];
ry(2.515637725974625) q[0];
ry(1.9351508274772362) q[2];
cx q[0],q[2];
ry(1.7407262615432169) q[2];
ry(-2.726574658156163) q[4];
cx q[2],q[4];
ry(0.17724410922389655) q[2];
ry(-0.1284093466765439) q[4];
cx q[2],q[4];
ry(0.13839613369416728) q[4];
ry(-0.03536301375195805) q[6];
cx q[4],q[6];
ry(-0.38966345608854086) q[4];
ry(0.00760907629788754) q[6];
cx q[4],q[6];
ry(-1.8092309498083512) q[6];
ry(3.044268782983804) q[8];
cx q[6],q[8];
ry(3.1285788384689366) q[6];
ry(-3.1293520274801634) q[8];
cx q[6],q[8];
ry(-1.7076320059026076) q[8];
ry(-1.757935331068846) q[10];
cx q[8],q[10];
ry(-0.003329856151820465) q[8];
ry(-3.1379255483390827) q[10];
cx q[8],q[10];
ry(1.146425281211953) q[10];
ry(1.675338219186922) q[12];
cx q[10],q[12];
ry(0.46401027951435436) q[10];
ry(0.017192400547974884) q[12];
cx q[10],q[12];
ry(-1.5585601065400398) q[12];
ry(-1.0211537911039734) q[14];
cx q[12],q[14];
ry(1.688791649701793) q[12];
ry(-2.036393403701279) q[14];
cx q[12],q[14];
ry(1.570897926939488) q[14];
ry(2.8866452617778102) q[16];
cx q[14],q[16];
ry(0.00023496639620823598) q[14];
ry(-3.140595925730967) q[16];
cx q[14],q[16];
ry(0.6329867498500185) q[16];
ry(-0.7168994596497242) q[18];
cx q[16],q[18];
ry(2.716987941000168) q[16];
ry(2.977033531495549) q[18];
cx q[16],q[18];
ry(-1.6293787322777353) q[1];
ry(1.6079477615860214) q[3];
cx q[1],q[3];
ry(-3.116971128166122) q[1];
ry(-0.05744977917709715) q[3];
cx q[1],q[3];
ry(-0.16420544854997388) q[3];
ry(2.114885147058703) q[5];
cx q[3],q[5];
ry(-3.100863113126732) q[3];
ry(2.3243894944145937) q[5];
cx q[3],q[5];
ry(0.9964638222627746) q[5];
ry(-1.580958028123099) q[7];
cx q[5],q[7];
ry(0.0581276526392457) q[5];
ry(0.003065792104793495) q[7];
cx q[5],q[7];
ry(-1.6143200538151614) q[7];
ry(1.0011053396985305) q[9];
cx q[7],q[9];
ry(0.10569040104939488) q[7];
ry(-2.5055898865532473) q[9];
cx q[7],q[9];
ry(-1.6991485281000944) q[9];
ry(-2.04618082065813) q[11];
cx q[9],q[11];
ry(-0.0026317713824497513) q[9];
ry(-0.00043127172578528343) q[11];
cx q[9],q[11];
ry(2.3187794704169358) q[11];
ry(2.383171687445768) q[13];
cx q[11],q[13];
ry(1.5495630410173844) q[11];
ry(-2.9333485774503574) q[13];
cx q[11],q[13];
ry(1.5606797353259552) q[13];
ry(-0.741558589289178) q[15];
cx q[13],q[15];
ry(0.3172187673515667) q[13];
ry(-1.1118087288720673) q[15];
cx q[13],q[15];
ry(-1.3575116299373986) q[15];
ry(-2.28539633100497) q[17];
cx q[15],q[17];
ry(5.145274743211868e-06) q[15];
ry(3.141539222032382) q[17];
cx q[15],q[17];
ry(0.6532622372530823) q[17];
ry(-2.7876115399851216) q[19];
cx q[17],q[19];
ry(-0.1891180122824352) q[17];
ry(2.870182787069813) q[19];
cx q[17],q[19];
ry(2.462136884632724) q[0];
ry(2.4589028345046167) q[3];
cx q[0],q[3];
ry(2.0619997713031744) q[0];
ry(-1.084421318718039) q[3];
cx q[0],q[3];
ry(1.4782010920385245) q[1];
ry(-1.7710756045730571) q[2];
cx q[1],q[2];
ry(1.854109998192115) q[1];
ry(-1.9433225901184497) q[2];
cx q[1],q[2];
ry(-2.438314301845535) q[2];
ry(2.2995597495872824) q[5];
cx q[2],q[5];
ry(0.35378573552071924) q[2];
ry(-0.9616804066414559) q[5];
cx q[2],q[5];
ry(1.4222571982317964) q[3];
ry(-1.1180351627532756) q[4];
cx q[3],q[4];
ry(-1.7005825825970577) q[3];
ry(-2.4204543668643295) q[4];
cx q[3],q[4];
ry(-2.288838499322059) q[4];
ry(2.2280552233405357) q[7];
cx q[4],q[7];
ry(3.137248042365398) q[4];
ry(-0.004213996500510931) q[7];
cx q[4],q[7];
ry(1.5517591284347323) q[5];
ry(-1.837816353994298) q[6];
cx q[5],q[6];
ry(-0.3671352246773685) q[5];
ry(-3.141090364929959) q[6];
cx q[5],q[6];
ry(1.7440028541667771) q[6];
ry(-1.953249848249465) q[9];
cx q[6],q[9];
ry(-0.08418520466496493) q[6];
ry(0.16855317608162856) q[9];
cx q[6],q[9];
ry(-1.8834542671769894) q[7];
ry(-2.271023173964024) q[8];
cx q[7],q[8];
ry(3.0784423299418457) q[7];
ry(-0.04600660476650287) q[8];
cx q[7],q[8];
ry(-0.34491063168910524) q[8];
ry(0.9725785719572271) q[11];
cx q[8],q[11];
ry(0.0015129147481536265) q[8];
ry(-0.001030114578502224) q[11];
cx q[8],q[11];
ry(-1.9215090946810909) q[9];
ry(0.9701827340022557) q[10];
cx q[9],q[10];
ry(3.1414421037392795) q[9];
ry(-2.233343944655633) q[10];
cx q[9],q[10];
ry(2.3991012275795427) q[10];
ry(0.7149562596572041) q[13];
cx q[10],q[13];
ry(1.3545821554007567) q[10];
ry(-1.7800637640168988) q[13];
cx q[10],q[13];
ry(1.5827099314648088) q[11];
ry(-0.4991056559858657) q[12];
cx q[11],q[12];
ry(1.606669828368716) q[11];
ry(-2.714018005011143) q[12];
cx q[11],q[12];
ry(-1.741355553478397) q[12];
ry(-2.8820854321149167) q[15];
cx q[12],q[15];
ry(2.63391459576549) q[12];
ry(0.6481981287089473) q[15];
cx q[12],q[15];
ry(-1.6072455299631105) q[13];
ry(0.7806275308543357) q[14];
cx q[13],q[14];
ry(0.024330898211771017) q[13];
ry(-3.100018333728585) q[14];
cx q[13],q[14];
ry(-2.3844862292441116) q[14];
ry(2.6350501291102417) q[17];
cx q[14],q[17];
ry(3.141564917315318) q[14];
ry(-3.141524255694555) q[17];
cx q[14],q[17];
ry(-2.667431750242014) q[15];
ry(1.3555449789920138) q[16];
cx q[15],q[16];
ry(3.1415663750837997) q[15];
ry(-0.00017608746126018812) q[16];
cx q[15],q[16];
ry(2.7259893811324174) q[16];
ry(-1.9193580654238824) q[19];
cx q[16],q[19];
ry(-2.7919067380645015) q[16];
ry(2.4581035751309077) q[19];
cx q[16],q[19];
ry(-2.6741883290396227) q[17];
ry(1.064409227899966) q[18];
cx q[17],q[18];
ry(2.3974194039255874) q[17];
ry(-2.517813448652979) q[18];
cx q[17],q[18];
ry(1.3802049763882847) q[0];
ry(1.6406631544824022) q[1];
cx q[0],q[1];
ry(1.6457515011786636) q[0];
ry(1.78622830966989) q[1];
cx q[0],q[1];
ry(-2.0667763266192933) q[2];
ry(0.26712701449886245) q[3];
cx q[2],q[3];
ry(1.3384875041496653) q[2];
ry(1.3851597669449411) q[3];
cx q[2],q[3];
ry(-1.3247562681701814) q[4];
ry(0.684905468782615) q[5];
cx q[4],q[5];
ry(-0.20052014714544883) q[4];
ry(-2.933782717226685) q[5];
cx q[4],q[5];
ry(1.541827969463823) q[6];
ry(-1.499463555105162) q[7];
cx q[6],q[7];
ry(-0.016432546520969815) q[6];
ry(-0.022266828743845224) q[7];
cx q[6],q[7];
ry(-2.73385849847302) q[8];
ry(2.717904630156061) q[9];
cx q[8],q[9];
ry(-2.6344695614471902) q[8];
ry(-0.7587682388441406) q[9];
cx q[8],q[9];
ry(-2.386935805120755) q[10];
ry(-0.8892034343999106) q[11];
cx q[10],q[11];
ry(-1.9557952393586566) q[10];
ry(0.8346002337715981) q[11];
cx q[10],q[11];
ry(1.5747583724749603) q[12];
ry(-3.0495787495979103) q[13];
cx q[12],q[13];
ry(2.501300476992474) q[12];
ry(0.6696892542075723) q[13];
cx q[12],q[13];
ry(0.9556517772213189) q[14];
ry(0.44154624048858004) q[15];
cx q[14],q[15];
ry(0.2768679252141757) q[14];
ry(-0.019321925405575688) q[15];
cx q[14],q[15];
ry(2.4418797093471922) q[16];
ry(1.8203128360652616) q[17];
cx q[16],q[17];
ry(1.0791665536697437) q[16];
ry(0.7010752229011112) q[17];
cx q[16],q[17];
ry(2.773239383123918) q[18];
ry(-0.045056316949398934) q[19];
cx q[18],q[19];
ry(1.4016318336429405) q[18];
ry(0.004490399215104289) q[19];
cx q[18],q[19];
ry(1.2715217112966224) q[0];
ry(1.1807137719206657) q[2];
cx q[0],q[2];
ry(3.034976902832945) q[0];
ry(-0.7015789993135035) q[2];
cx q[0],q[2];
ry(2.489033021043075) q[2];
ry(-1.1792890939575944) q[4];
cx q[2],q[4];
ry(0.8713507241760087) q[2];
ry(2.66084599766486) q[4];
cx q[2],q[4];
ry(2.612348129001716) q[4];
ry(3.085526223513274) q[6];
cx q[4],q[6];
ry(0.015142941542264053) q[4];
ry(3.1415876558804903) q[6];
cx q[4],q[6];
ry(1.5022651070698476) q[6];
ry(-0.5020320615736233) q[8];
cx q[6],q[8];
ry(0.056036292379356485) q[6];
ry(-1.9656817072110941) q[8];
cx q[6],q[8];
ry(-2.876887182012201) q[8];
ry(-0.8823928996680455) q[10];
cx q[8],q[10];
ry(-0.003167546764775153) q[8];
ry(-0.0012949648047271111) q[10];
cx q[8],q[10];
ry(-0.6947643103446915) q[10];
ry(-0.7688941495016852) q[12];
cx q[10],q[12];
ry(0.9324943106029432) q[10];
ry(1.860605269007832) q[12];
cx q[10],q[12];
ry(-3.133645647598961) q[12];
ry(2.1306995595900804) q[14];
cx q[12],q[14];
ry(3.0148426679832707) q[12];
ry(-3.1009433744378687) q[14];
cx q[12],q[14];
ry(1.8506178435750185) q[14];
ry(-1.5434260734828107) q[16];
cx q[14],q[16];
ry(-8.812504488542316e-05) q[14];
ry(-0.0003480256767836125) q[16];
cx q[14],q[16];
ry(-1.7889971099550441) q[16];
ry(0.3997373961734898) q[18];
cx q[16],q[18];
ry(-2.230167200400415) q[16];
ry(-2.074274418622906) q[18];
cx q[16],q[18];
ry(-0.15995929974811673) q[1];
ry(-1.1704308394514136) q[3];
cx q[1],q[3];
ry(-0.5980882964515803) q[1];
ry(2.0653073107824165) q[3];
cx q[1],q[3];
ry(0.8689919176731267) q[3];
ry(-1.5979549476059969) q[5];
cx q[3],q[5];
ry(0.7915339831026964) q[3];
ry(2.1576927460480304) q[5];
cx q[3],q[5];
ry(1.2974919181455626) q[5];
ry(-0.36272447276962033) q[7];
cx q[5],q[7];
ry(-0.07903442186280694) q[5];
ry(3.00398709410557) q[7];
cx q[5],q[7];
ry(1.6104364973720589) q[7];
ry(-0.24302239414600194) q[9];
cx q[7],q[9];
ry(-2.6381252042218155) q[7];
ry(-2.7430968271810996) q[9];
cx q[7],q[9];
ry(-2.1178228538466257) q[9];
ry(-3.008069483195711) q[11];
cx q[9],q[11];
ry(-3.1412620018084905) q[9];
ry(-0.004611914594411459) q[11];
cx q[9],q[11];
ry(0.7781741764362583) q[11];
ry(-1.2982656978994225) q[13];
cx q[11],q[13];
ry(-1.3893404576704909) q[11];
ry(-0.557780736252183) q[13];
cx q[11],q[13];
ry(0.7540774735382301) q[13];
ry(2.3512613518172243) q[15];
cx q[13],q[15];
ry(-0.22137875383831968) q[13];
ry(-2.330483543870194) q[15];
cx q[13],q[15];
ry(-0.7443048922604846) q[15];
ry(3.1065275361239704) q[17];
cx q[15],q[17];
ry(-0.0005459523496851304) q[15];
ry(-3.1412540958278488) q[17];
cx q[15],q[17];
ry(3.0270317329465324) q[17];
ry(1.3228405409722461) q[19];
cx q[17],q[19];
ry(2.651040562430426) q[17];
ry(2.9512381697871617) q[19];
cx q[17],q[19];
ry(0.1418205573929107) q[0];
ry(-2.9959429151102075) q[3];
cx q[0],q[3];
ry(3.115308750644541) q[0];
ry(3.030568301278889) q[3];
cx q[0],q[3];
ry(1.4893442861115613) q[1];
ry(-0.2934400550918023) q[2];
cx q[1],q[2];
ry(-1.7325775316793162) q[1];
ry(-2.900989239848934) q[2];
cx q[1],q[2];
ry(0.03569632946128909) q[2];
ry(2.615912872657264) q[5];
cx q[2],q[5];
ry(-3.1399151596439956) q[2];
ry(3.1395284831413686) q[5];
cx q[2],q[5];
ry(1.6389683572999934) q[3];
ry(-0.45793061081323305) q[4];
cx q[3],q[4];
ry(1.7469126880160653) q[3];
ry(-1.1602417686856676) q[4];
cx q[3],q[4];
ry(0.9237718864184297) q[4];
ry(1.6706666667089494) q[7];
cx q[4],q[7];
ry(-3.1371197968169704) q[4];
ry(0.003078539577280992) q[7];
cx q[4],q[7];
ry(2.387036247770756) q[5];
ry(1.9387332927442484) q[6];
cx q[5],q[6];
ry(0.0051017352396938165) q[5];
ry(-3.1360900358530843) q[6];
cx q[5],q[6];
ry(1.1347564215248669) q[6];
ry(2.3396139534095446) q[9];
cx q[6],q[9];
ry(-3.0056489514861426) q[6];
ry(1.0180279556978045) q[9];
cx q[6],q[9];
ry(-2.9344258203294977) q[7];
ry(1.7199375495585691) q[8];
cx q[7],q[8];
ry(-3.083081316590826) q[7];
ry(-0.02806312726744409) q[8];
cx q[7],q[8];
ry(-1.5852344005789094) q[8];
ry(-2.7500108327323907) q[11];
cx q[8],q[11];
ry(0.006385229019257765) q[8];
ry(3.1358638826314236) q[11];
cx q[8],q[11];
ry(1.239000484088801) q[9];
ry(0.5353499493698495) q[10];
cx q[9],q[10];
ry(-3.1408429614834272) q[9];
ry(-0.002847644604550422) q[10];
cx q[9],q[10];
ry(-0.20991856646850326) q[10];
ry(-1.3418955894189355) q[13];
cx q[10],q[13];
ry(-3.0667370144108497) q[10];
ry(0.34696791211485145) q[13];
cx q[10],q[13];
ry(-1.8300912481444553) q[11];
ry(2.7360994942650514) q[12];
cx q[11],q[12];
ry(-2.5269952656272094) q[11];
ry(-0.5440154067744754) q[12];
cx q[11],q[12];
ry(-0.29018709758967987) q[12];
ry(-2.2988451902836746) q[15];
cx q[12],q[15];
ry(-2.9359033344672723) q[12];
ry(0.0054537514822703415) q[15];
cx q[12],q[15];
ry(1.2227188195694554) q[13];
ry(0.9929802576876527) q[14];
cx q[13],q[14];
ry(-2.977717470011025) q[13];
ry(-0.2047573717674297) q[14];
cx q[13],q[14];
ry(1.1443631935700411) q[14];
ry(-2.8700999311465423) q[17];
cx q[14],q[17];
ry(-6.294541699869428e-05) q[14];
ry(3.141531730624441) q[17];
cx q[14],q[17];
ry(-2.314412738186633) q[15];
ry(1.3643920871519346) q[16];
cx q[15],q[16];
ry(-3.138927325514255) q[15];
ry(-0.0017427347907070546) q[16];
cx q[15],q[16];
ry(1.8699006661655382) q[16];
ry(2.744404871458968) q[19];
cx q[16],q[19];
ry(0.17419171473077652) q[16];
ry(0.1288240550897437) q[19];
cx q[16],q[19];
ry(-2.6148413375507684) q[17];
ry(2.3333638023258763) q[18];
cx q[17],q[18];
ry(-2.117460431133912) q[17];
ry(0.37654450308202686) q[18];
cx q[17],q[18];
ry(-0.26173559837048604) q[0];
ry(-1.6837328409638062) q[1];
cx q[0],q[1];
ry(-2.6825978445605774) q[0];
ry(-0.3352644864488931) q[1];
cx q[0],q[1];
ry(0.640647406095435) q[2];
ry(3.073395486942532) q[3];
cx q[2],q[3];
ry(1.6172544852829651) q[2];
ry(1.5197332300640136) q[3];
cx q[2],q[3];
ry(-2.389893030955499) q[4];
ry(-0.12676167247839665) q[5];
cx q[4],q[5];
ry(-3.0042547929270276) q[4];
ry(-0.012657770100308028) q[5];
cx q[4],q[5];
ry(0.2057938805060213) q[6];
ry(2.633653615651938) q[7];
cx q[6],q[7];
ry(0.068369455221009) q[6];
ry(-1.7252344608422678) q[7];
cx q[6],q[7];
ry(-2.5735596812698818) q[8];
ry(0.5937035581140018) q[9];
cx q[8],q[9];
ry(1.4708321826807969) q[8];
ry(1.6164779797773292) q[9];
cx q[8],q[9];
ry(0.9080828891154678) q[10];
ry(1.6763271610020745) q[11];
cx q[10],q[11];
ry(-1.9257227638621996) q[10];
ry(-2.0813914026329843) q[11];
cx q[10],q[11];
ry(1.8175440981198705) q[12];
ry(-0.4090059753009925) q[13];
cx q[12],q[13];
ry(-2.6411715809282974) q[12];
ry(1.2355084871833224) q[13];
cx q[12],q[13];
ry(-2.5151784303715323) q[14];
ry(-2.8714235938187977) q[15];
cx q[14],q[15];
ry(-0.6092545618461642) q[14];
ry(0.2852449124030647) q[15];
cx q[14],q[15];
ry(-2.5600550003390348) q[16];
ry(-0.9035120287567928) q[17];
cx q[16],q[17];
ry(0.2861638116962837) q[16];
ry(2.9175394820698273) q[17];
cx q[16],q[17];
ry(0.3530100038860917) q[18];
ry(1.1497734148390357) q[19];
cx q[18],q[19];
ry(-2.049622738613711) q[18];
ry(-3.0834824570062227) q[19];
cx q[18],q[19];
ry(-1.4988810674522401) q[0];
ry(0.9740432953192135) q[2];
cx q[0],q[2];
ry(2.7203556140026506) q[0];
ry(2.975630484509866) q[2];
cx q[0],q[2];
ry(-0.4199628737762815) q[2];
ry(-2.98541035030653) q[4];
cx q[2],q[4];
ry(-0.047278587993305976) q[2];
ry(0.029832879919701405) q[4];
cx q[2],q[4];
ry(2.9331590049742187) q[4];
ry(-1.88309165432232) q[6];
cx q[4],q[6];
ry(3.136539600968739) q[4];
ry(3.1393676996240742) q[6];
cx q[4],q[6];
ry(1.114771441894173) q[6];
ry(-0.0024256051426299052) q[8];
cx q[6],q[8];
ry(3.0699181804294087) q[6];
ry(1.5351093896579384) q[8];
cx q[6],q[8];
ry(-2.1997275629192647) q[8];
ry(-2.5250533625630083) q[10];
cx q[8],q[10];
ry(-3.141307844473985) q[8];
ry(-0.052253873357536076) q[10];
cx q[8],q[10];
ry(-1.4988955827374157) q[10];
ry(-2.0435064760510393) q[12];
cx q[10],q[12];
ry(-2.6882129856207415) q[10];
ry(0.02652897068007487) q[12];
cx q[10],q[12];
ry(-2.93524548398653) q[12];
ry(1.891194436283433) q[14];
cx q[12],q[14];
ry(-3.1052852089110865) q[12];
ry(-3.120091234727736) q[14];
cx q[12],q[14];
ry(1.8320474095786707) q[14];
ry(-1.8946583569829576) q[16];
cx q[14],q[16];
ry(-3.1385987058345544) q[14];
ry(0.0015871471123768553) q[16];
cx q[14],q[16];
ry(-1.4139938378054748) q[16];
ry(-2.2318312717280238) q[18];
cx q[16],q[18];
ry(3.1166813080678373) q[16];
ry(-0.023948983902426054) q[18];
cx q[16],q[18];
ry(-2.9591564046817984) q[1];
ry(-1.0668147118844862) q[3];
cx q[1],q[3];
ry(2.5913639335305483) q[1];
ry(-0.7736488264537966) q[3];
cx q[1],q[3];
ry(-0.3072705825112978) q[3];
ry(1.2952615500092843) q[5];
cx q[3],q[5];
ry(-3.1377373199393745) q[3];
ry(3.1302700583883496) q[5];
cx q[3],q[5];
ry(-0.32271924338733804) q[5];
ry(2.8894278182991493) q[7];
cx q[5],q[7];
ry(0.03483185022952995) q[5];
ry(0.005442569408753543) q[7];
cx q[5],q[7];
ry(-0.12018596329035931) q[7];
ry(-2.6243839054613662) q[9];
cx q[7],q[9];
ry(-3.1123669386140094) q[7];
ry(-3.085924994482628) q[9];
cx q[7],q[9];
ry(-1.4174166373959778) q[9];
ry(-0.43394981732546345) q[11];
cx q[9],q[11];
ry(3.088681577753324) q[9];
ry(3.104747425004164) q[11];
cx q[9],q[11];
ry(1.387327100654896) q[11];
ry(0.6856715392547507) q[13];
cx q[11],q[13];
ry(0.03625216440634743) q[11];
ry(0.014386718244577601) q[13];
cx q[11],q[13];
ry(0.7794001612846664) q[13];
ry(2.8797170717555005) q[15];
cx q[13],q[15];
ry(1.2940978867786552) q[13];
ry(-2.875489569554141) q[15];
cx q[13],q[15];
ry(-1.3541469863131288) q[15];
ry(2.4868735952542043) q[17];
cx q[15],q[17];
ry(-3.1400538713481025) q[15];
ry(-3.1413011392310644) q[17];
cx q[15],q[17];
ry(1.0162385190348955) q[17];
ry(3.0281863547516967) q[19];
cx q[17],q[19];
ry(0.04783680979761762) q[17];
ry(0.46056779430358735) q[19];
cx q[17],q[19];
ry(-0.8236349647548211) q[0];
ry(0.1517725154693537) q[3];
cx q[0],q[3];
ry(1.4817158915588857) q[0];
ry(1.4126340473405157) q[3];
cx q[0],q[3];
ry(-2.320484721157732) q[1];
ry(-2.622214150758082) q[2];
cx q[1],q[2];
ry(0.23916719507864906) q[1];
ry(-1.038985484693724) q[2];
cx q[1],q[2];
ry(-1.8953015223883023) q[2];
ry(-1.044575254723811) q[5];
cx q[2],q[5];
ry(-0.028622195511737836) q[2];
ry(-3.1410798132429876) q[5];
cx q[2],q[5];
ry(-2.5479674798415703) q[3];
ry(-1.6769890742633153) q[4];
cx q[3],q[4];
ry(-3.130682300766115) q[3];
ry(3.129450855525811) q[4];
cx q[3],q[4];
ry(2.619237694024056) q[4];
ry(-0.9255315148185188) q[7];
cx q[4],q[7];
ry(-0.0043572404556364575) q[4];
ry(0.027427374717926504) q[7];
cx q[4],q[7];
ry(1.5960910685533802) q[5];
ry(-0.6393884991585885) q[6];
cx q[5],q[6];
ry(-0.05864298641670729) q[5];
ry(0.01939146107062406) q[6];
cx q[5],q[6];
ry(-1.6371804312324443) q[6];
ry(1.7948660941028827) q[9];
cx q[6],q[9];
ry(9.503530927545967e-05) q[6];
ry(-0.024711282929605005) q[9];
cx q[6],q[9];
ry(2.5104938088318787) q[7];
ry(-0.9071186666151991) q[8];
cx q[7],q[8];
ry(-0.2050391247379506) q[7];
ry(0.026172674260294102) q[8];
cx q[7],q[8];
ry(-1.5874315139291522) q[8];
ry(1.6700880342269007) q[11];
cx q[8],q[11];
ry(-3.1210297936473173) q[8];
ry(0.04737942068165843) q[11];
cx q[8],q[11];
ry(0.8457985738427398) q[9];
ry(-0.571260926078919) q[10];
cx q[9],q[10];
ry(-0.010903072689211868) q[9];
ry(-2.9119889871527103) q[10];
cx q[9],q[10];
ry(-0.7602452523515248) q[10];
ry(-1.535854498864155) q[13];
cx q[10],q[13];
ry(0.1863465906053019) q[10];
ry(0.005699797088092318) q[13];
cx q[10],q[13];
ry(2.173101987491) q[11];
ry(-2.70697014195892) q[12];
cx q[11],q[12];
ry(-3.1385392350943246) q[11];
ry(3.1222920600394084) q[12];
cx q[11],q[12];
ry(-0.07436923869557166) q[12];
ry(-0.6629232729507439) q[15];
cx q[12],q[15];
ry(0.1828650559870466) q[12];
ry(2.763224713745194) q[15];
cx q[12],q[15];
ry(0.40434763356651615) q[13];
ry(2.6851669804880944) q[14];
cx q[13],q[14];
ry(0.571180442032075) q[13];
ry(-0.3783591318473278) q[14];
cx q[13],q[14];
ry(-2.1353533117909267) q[14];
ry(2.4051011186330573) q[17];
cx q[14],q[17];
ry(3.1412397685254674) q[14];
ry(1.3376183748192716e-05) q[17];
cx q[14],q[17];
ry(-0.6550589062411589) q[15];
ry(-2.211028883083975) q[16];
cx q[15],q[16];
ry(2.958048498601348) q[15];
ry(2.7754350403924706) q[16];
cx q[15],q[16];
ry(-1.9103279335824699) q[16];
ry(1.1814418154388093) q[19];
cx q[16],q[19];
ry(3.067740100479573) q[16];
ry(-3.122186956708852) q[19];
cx q[16],q[19];
ry(-0.4918971534932862) q[17];
ry(-2.4865370976021657) q[18];
cx q[17],q[18];
ry(2.8016789890203673) q[17];
ry(-2.6983895180520867) q[18];
cx q[17],q[18];
ry(-1.0868832603841065) q[0];
ry(2.269471744649703) q[1];
cx q[0],q[1];
ry(-2.152392937692773) q[0];
ry(-1.3936010171483386) q[1];
cx q[0],q[1];
ry(-0.05037042462654995) q[2];
ry(-0.7652653134275834) q[3];
cx q[2],q[3];
ry(-1.3928744551609342) q[2];
ry(-0.08954882271535958) q[3];
cx q[2],q[3];
ry(1.3464004642437342) q[4];
ry(2.1496269373278976) q[5];
cx q[4],q[5];
ry(-1.9135843622999846) q[4];
ry(1.128540211351818) q[5];
cx q[4],q[5];
ry(0.08638038525793412) q[6];
ry(-1.2124335191845876) q[7];
cx q[6],q[7];
ry(2.760854821993479) q[6];
ry(-0.22861198916145753) q[7];
cx q[6],q[7];
ry(-0.5051112633559312) q[8];
ry(0.8894393102584841) q[9];
cx q[8],q[9];
ry(0.0728113697950499) q[8];
ry(0.07962964143916729) q[9];
cx q[8],q[9];
ry(-1.787698397935264) q[10];
ry(-2.8656109379468653) q[11];
cx q[10],q[11];
ry(2.6820394432120485) q[10];
ry(0.1729412892372064) q[11];
cx q[10],q[11];
ry(0.34407974899877847) q[12];
ry(-1.1373406713930976) q[13];
cx q[12],q[13];
ry(2.2707787311415393) q[12];
ry(-3.0948472431977163) q[13];
cx q[12],q[13];
ry(0.6131658943525116) q[14];
ry(-1.1693412260032847) q[15];
cx q[14],q[15];
ry(3.1386411223208692) q[14];
ry(0.002770140868733506) q[15];
cx q[14],q[15];
ry(2.427989446984292) q[16];
ry(-2.721796695745681) q[17];
cx q[16],q[17];
ry(-0.20388078167704382) q[16];
ry(0.47680452388543504) q[17];
cx q[16],q[17];
ry(1.6203626474250303) q[18];
ry(-1.767164151450495) q[19];
cx q[18],q[19];
ry(0.15994161334546142) q[18];
ry(1.1219728157222695) q[19];
cx q[18],q[19];
ry(1.2171504285876855) q[0];
ry(-2.0104207699211685) q[2];
cx q[0],q[2];
ry(0.18025193075581036) q[0];
ry(0.0897724020728946) q[2];
cx q[0],q[2];
ry(1.8604822272080463) q[2];
ry(-2.5464791853242006) q[4];
cx q[2],q[4];
ry(-3.123840692844456) q[2];
ry(0.059694290362182115) q[4];
cx q[2],q[4];
ry(-3.039515587145204) q[4];
ry(-2.5826203586193874) q[6];
cx q[4],q[6];
ry(-0.005009332675478539) q[4];
ry(-0.14714850740968663) q[6];
cx q[4],q[6];
ry(0.2633554218845138) q[6];
ry(1.3333055744538242) q[8];
cx q[6],q[8];
ry(0.04422133167228184) q[6];
ry(0.005681407790609411) q[8];
cx q[6],q[8];
ry(-1.7824398644084578) q[8];
ry(2.147512372817669) q[10];
cx q[8],q[10];
ry(-3.105798068287452) q[8];
ry(0.008916946779357874) q[10];
cx q[8],q[10];
ry(-2.075628490684232) q[10];
ry(-1.6579329020261717) q[12];
cx q[10],q[12];
ry(-3.0871558091532196) q[10];
ry(-0.08900373665769834) q[12];
cx q[10],q[12];
ry(-0.6559232545101082) q[12];
ry(-2.451230169251986) q[14];
cx q[12],q[14];
ry(-2.557182972948192) q[12];
ry(0.4047223620552855) q[14];
cx q[12],q[14];
ry(2.2368486929256024) q[14];
ry(1.289101528491831) q[16];
cx q[14],q[16];
ry(3.1303526375619195) q[14];
ry(3.1262557020083577) q[16];
cx q[14],q[16];
ry(1.2934380708747293) q[16];
ry(0.808057417968043) q[18];
cx q[16],q[18];
ry(0.014695462387601665) q[16];
ry(-3.115539976911853) q[18];
cx q[16],q[18];
ry(-3.1047150482166477) q[1];
ry(-1.784067468021746) q[3];
cx q[1],q[3];
ry(1.0199539778131559) q[1];
ry(-1.1458577679282422) q[3];
cx q[1],q[3];
ry(-1.0474727382362525) q[3];
ry(2.7269737023192975) q[5];
cx q[3],q[5];
ry(-0.002560646619462845) q[3];
ry(-3.139353076877707) q[5];
cx q[3],q[5];
ry(1.219825752982869) q[5];
ry(1.2090395176901017) q[7];
cx q[5],q[7];
ry(0.0015514368437905546) q[5];
ry(-0.003948010293418265) q[7];
cx q[5],q[7];
ry(-2.4090243306407713) q[7];
ry(-1.5368450744043691) q[9];
cx q[7],q[9];
ry(-3.1339901657016447) q[7];
ry(-3.136894008699113) q[9];
cx q[7],q[9];
ry(0.5092676605396527) q[9];
ry(1.9202757449972632) q[11];
cx q[9],q[11];
ry(3.1151779877084054) q[9];
ry(0.012834217396812342) q[11];
cx q[9],q[11];
ry(2.8961842252883647) q[11];
ry(2.5765310389240743) q[13];
cx q[11],q[13];
ry(0.08953627858665048) q[11];
ry(-3.138016811372311) q[13];
cx q[11],q[13];
ry(-0.6897096193350771) q[13];
ry(1.0408327712768601) q[15];
cx q[13],q[15];
ry(0.004570642386856831) q[13];
ry(-3.140012304098134) q[15];
cx q[13],q[15];
ry(1.6679318036405586) q[15];
ry(2.189160258088586) q[17];
cx q[15],q[17];
ry(-3.131358366119317) q[15];
ry(-0.0020003257241807404) q[17];
cx q[15],q[17];
ry(0.8984642255438217) q[17];
ry(1.8666573313355994) q[19];
cx q[17],q[19];
ry(3.003461789443854) q[17];
ry(-3.107119521035211) q[19];
cx q[17],q[19];
ry(3.0407087789582183) q[0];
ry(0.1268103657261319) q[3];
cx q[0],q[3];
ry(-1.399636294177481) q[0];
ry(-2.1224398963864735) q[3];
cx q[0],q[3];
ry(0.35585854338378914) q[1];
ry(1.180235753703328) q[2];
cx q[1],q[2];
ry(0.49047356671677317) q[1];
ry(-0.22884590479789768) q[2];
cx q[1],q[2];
ry(2.5007191683409404) q[2];
ry(2.7200696559139153) q[5];
cx q[2],q[5];
ry(3.1179144350221804) q[2];
ry(0.00844651377334138) q[5];
cx q[2],q[5];
ry(0.9047606189278321) q[3];
ry(1.666022308392678) q[4];
cx q[3],q[4];
ry(-0.3488636281609976) q[3];
ry(3.0077533429315326) q[4];
cx q[3],q[4];
ry(1.6148043596671633) q[4];
ry(-0.36638750759956823) q[7];
cx q[4],q[7];
ry(-3.1391675014502387) q[4];
ry(-3.139784040631927) q[7];
cx q[4],q[7];
ry(1.8542217008882362) q[5];
ry(1.4109795773035323) q[6];
cx q[5],q[6];
ry(0.35718795260422453) q[5];
ry(3.035350075770903) q[6];
cx q[5],q[6];
ry(-2.7072128437959075) q[6];
ry(1.5716362413747937) q[9];
cx q[6],q[9];
ry(3.1410366137967833) q[6];
ry(0.0002367500248423582) q[9];
cx q[6],q[9];
ry(-1.916463207892022) q[7];
ry(-1.9485818235769004) q[8];
cx q[7],q[8];
ry(3.017416620516578) q[7];
ry(0.0540169848835354) q[8];
cx q[7],q[8];
ry(-1.8724596124158503) q[8];
ry(-1.3172326071405935) q[11];
cx q[8],q[11];
ry(-0.0019791954563368463) q[8];
ry(-0.0013688432025640296) q[11];
cx q[8],q[11];
ry(0.2583274331901731) q[9];
ry(-0.16755758638387785) q[10];
cx q[9],q[10];
ry(3.0854614938751825) q[9];
ry(0.01016833748449126) q[10];
cx q[9],q[10];
ry(-0.24016055519169033) q[10];
ry(-2.898307999088733) q[13];
cx q[10],q[13];
ry(0.019084434343233897) q[10];
ry(3.0726282102950035) q[13];
cx q[10],q[13];
ry(-1.4070806511568712) q[11];
ry(-3.08872160776795) q[12];
cx q[11],q[12];
ry(3.0765690275155273) q[11];
ry(-2.9146298615251243) q[12];
cx q[11],q[12];
ry(2.543067617747269) q[12];
ry(-0.10387195161316283) q[15];
cx q[12],q[15];
ry(-0.001419071293879526) q[12];
ry(-3.139212410342479) q[15];
cx q[12],q[15];
ry(-2.679051265569398) q[13];
ry(0.0072869320452673145) q[14];
cx q[13],q[14];
ry(-0.051997740672171854) q[13];
ry(-0.06022990459512731) q[14];
cx q[13],q[14];
ry(0.07723304150766458) q[14];
ry(1.0410875893386224) q[17];
cx q[14],q[17];
ry(-3.137030525389911) q[14];
ry(3.135656107528675) q[17];
cx q[14],q[17];
ry(1.9524414609294585) q[15];
ry(1.6500663077449202) q[16];
cx q[15],q[16];
ry(3.0848605948010186) q[15];
ry(0.01394404075618727) q[16];
cx q[15],q[16];
ry(-1.6138578216498152) q[16];
ry(-1.5481209391024708) q[19];
cx q[16],q[19];
ry(-0.05599424850149237) q[16];
ry(-0.10850934291792137) q[19];
cx q[16],q[19];
ry(-1.5884026056098113) q[17];
ry(0.540688331004726) q[18];
cx q[17],q[18];
ry(-0.29406075114169095) q[17];
ry(-0.3949752286338387) q[18];
cx q[17],q[18];
ry(-0.2845185555975652) q[0];
ry(2.436514306743859) q[1];
ry(2.175482724574764) q[2];
ry(0.0217531444065847) q[3];
ry(0.044932452708292984) q[4];
ry(3.1267724105770953) q[5];
ry(1.1377093806224332) q[6];
ry(1.1093595390572293) q[7];
ry(-0.7841555599254937) q[8];
ry(-0.8212915495436564) q[9];
ry(0.00851306219459236) q[10];
ry(0.09430011807502149) q[11];
ry(-0.6169289372839044) q[12];
ry(-0.02894587381105485) q[13];
ry(0.07879063252665652) q[14];
ry(-2.60078739605481) q[15];
ry(0.08204124782552125) q[16];
ry(-0.045134976663519843) q[17];
ry(2.8440438091481792) q[18];
ry(0.7617549963637996) q[19];