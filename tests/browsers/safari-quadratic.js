// Copyright 2024, University of Colorado Boulder

/* eslint-disable */
// TODO: fix lint https://github.com/phetsims/chipper/issues/1405

/**
 * For reproduction using kite playground, see https://github.com/phetsims/graphing-quadratics/issues/206
 */

// window.innerWIdth: 1012, window.innerHeight 823 reproducible

const matrix = phet.dot.Matrix3.rowMajor(
  26.5, 0, 345,
  0, -26.5, 330,
  0, 0, 1
);

const minX = -10;
const maxX = 10;
const length = maxX - minX;

const getFromA = a => {
  const aPrime = a * length * length;
  const bPrime = 2 * a * minX * length;
  const cPrime = a * minX * minX;

  return new phet.kite.Shape()
    .moveToPoint(
      new phet.dot.Vector2( minX, cPrime )
    )
    .quadraticCurveToPoint(
      new phet.dot.Vector2( ( minX + maxX ) / 2, bPrime / 2 + cPrime ),
      new phet.dot.Vector2( maxX, aPrime + bPrime + cPrime )
    )
    .transformed( matrix );
};

const getSVGString = shape => `<path d="${shape.getSVGPath().replaceAll( '.00000000000000000000 ', ' ' ).trim()}" style="fill: blue; stroke: red; stroke-width: 10;"></path>\n`;

const basicShape = getFromA( 1 );
const quadraticSegment = basicShape.subpaths[ 0 ].segments[ 0 ];
const elevatedCubicSegment = quadraticSegment.degreeElevated();
console.log( elevatedCubicSegment.getSVGPathFragment() );
console.log( quadraticSegment.subdivided( 0.5 )[ 0 ].getSVGPathFragment() );
console.log( quadraticSegment.subdivided( 0.5 )[ 1 ].getSVGPathFragment() );

const getReproductionLine = a => `<!-- ${a} -->\n  ${getSVGString( getFromA( a ) )}\n`;

// console.log( getReproductionLine( 1 ) );

//copy( _.range( 0.6, 1.2, 0.01 ).map( a => getReproductionLine( a ) ).join( '' ) );



const getSmallLine = a => {
  return `<!-- ${a} -->\n  <path d="${getFromA( a ).getSVGPath().replaceAll( '.00000000000000000000 ', ' ' ).trim()}" style="fill: none; stroke: rgba(0,0,0,0.2); stroke-width: 0.2;"></path>\n`;
};

copy( _.range( 0.2, 3, 0.001 ).map( a => getSmallLine( a ) ).join( '' ) );
