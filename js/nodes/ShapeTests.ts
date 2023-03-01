// Copyright 2017-2023, University of Colorado Boulder

/**
 * Shape tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { LineStyles, Shape } from '../../../kite/js/imports.js';
import snapshotEquals from '../tests/snapshotEquals.js';
import Node from './Node.js';
import Path from './Path.js';

QUnit.module( 'Shape' );

const canvasWidth = 320;
const canvasHeight = 240;

// takes a snapshot of a scene and stores the pixel data, so that we can compare them
function snapshot( scene: Node, width?: number, height?: number ): ImageData {

  width = width || canvasWidth;
  height = height || canvasHeight;

  const canvas = document.createElement( 'canvas' );
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext( '2d' )!;
  scene.renderToCanvas( canvas, context );
  const data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
  return data;
}

// TODO: factor out
function sceneEquals( assert: Assert, constructionA: ( node: Node ) => void, constructionB: ( node: Node ) => void,
                      message?: string, threshold?: number ): boolean {

  if ( threshold === undefined ) {
    threshold = 0;
  }

  const sceneA = new Node();
  const sceneB = new Node();

  constructionA( sceneA );
  constructionB( sceneB );

  // sceneA.renderScene();
  // sceneB.renderScene();

  const isEqual = snapshotEquals( assert, snapshot( sceneA ), snapshot( sceneB ), threshold, message );

  // TODO: consider showing if tests fail
  return isEqual;
}

// TODO: factor out
function strokeEqualsFill( assert: Assert, shapeToStroke: Shape, shapeToFill: Shape,
                           strokeNodeSetup?: ( node: Path ) => void, message?: string ): void {

  sceneEquals( assert, scene => {
    const node = new Path( null );
    node.setShape( shapeToStroke );
    node.setStroke( '#000000' );
    if ( strokeNodeSetup ) { strokeNodeSetup( node ); }
    scene.addChild( node );
  }, scene => {
    const node = new Path( null );
    node.setShape( shapeToFill );
    node.setFill( '#000000' );
    // node.setStroke( '#ff0000' ); // for debugging strokes
    scene.addChild( node );
    // node.validateBounds();
    // scene.addChild( new Path( {
    //   shape: phet.kite.Shape.bounds( node.getSelfBounds() ),
    //   fill: 'rgba(0,0,255,0.5)'
    // } ) );
  }, message, 128 ); // threshold of 128 due to antialiasing differences between fill and stroke... :(
}

function p( x: number, y: number ): Vector2 { return new Vector2( x, y ); }

QUnit.test( 'Verifying Line/Rect', assert => {
  const lineWidth = 50;
  // /shapeToStroke, shapeToFill, strokeNodeSetup, message, debugFlag
  const strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
  const fillShape = Shape.rectangle( 100, 100 - lineWidth / 2, 200, lineWidth );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineWidth( lineWidth ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Segment - butt', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 50;

  const strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Segment - square', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 50;
  styles.lineCap = 'square';

  const strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Segment - round', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 50;
  styles.lineCap = 'round';

  const strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Join - Miter', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;
  styles.lineJoin = 'miter';

  const strokeShape = new Shape();
  strokeShape.moveTo( 70, 70 );
  strokeShape.lineTo( 140, 200 );
  strokeShape.lineTo( 210, 70 );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Join - Miter - Closed', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;
  styles.lineJoin = 'miter';

  const strokeShape = new Shape();
  strokeShape.moveTo( 70, 70 );
  strokeShape.lineTo( 140, 200 );
  strokeShape.lineTo( 210, 70 );
  strokeShape.close();
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Join - Round', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;
  styles.lineJoin = 'round';

  const strokeShape = new Shape();
  strokeShape.moveTo( 70, 70 );
  strokeShape.lineTo( 140, 200 );
  strokeShape.lineTo( 210, 70 );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Join - Round - Closed', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;
  styles.lineJoin = 'round';

  const strokeShape = new Shape();
  strokeShape.moveTo( 70, 70 );
  strokeShape.lineTo( 140, 200 );
  strokeShape.lineTo( 210, 70 );
  strokeShape.close();
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Line Join - Bevel - Closed', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;
  styles.lineJoin = 'bevel';

  const strokeShape = new Shape();
  strokeShape.moveTo( 70, 70 );
  strokeShape.lineTo( 140, 200 );
  strokeShape.lineTo( 210, 70 );
  strokeShape.close();
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Rect', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;

  const strokeShape = Shape.rectangle( 40, 40, 150, 150 );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Manual Rect', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;

  const strokeShape = new Shape();
  strokeShape.moveTo( 40, 40 );
  strokeShape.lineTo( 190, 40 );
  strokeShape.lineTo( 190, 190 );
  strokeShape.lineTo( 40, 190 );
  strokeShape.lineTo( 40, 40 );
  strokeShape.close();
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Hex', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;

  const strokeShape = Shape.regularPolygon( 6, 100 ).transformed( Matrix3.translation( 130, 130 ) );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Overlap', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;

  const strokeShape = new Shape();
  strokeShape.moveTo( 40, 40 );
  strokeShape.lineTo( 200, 200 );
  strokeShape.lineTo( 40, 200 );
  strokeShape.lineTo( 200, 40 );
  strokeShape.lineTo( 100, 0 );
  strokeShape.close();
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

const miterMagnitude = 160;
const miterAnglesInDegrees = [ 5, 8, 10, 11.5, 13, 20, 24, 30, 45 ];

_.each( miterAnglesInDegrees, miterAngle => {
  const miterAngleRadians = miterAngle * Math.PI / 180;
  QUnit.test( `Miter limit angle (degrees): ${miterAngle} would change at ${1 / Math.sin( miterAngleRadians / 2 )}`, assert => {
    const styles = new LineStyles();
    styles.lineWidth = 30;

    const strokeShape = new Shape();
    let point = new Vector2( 40, 100 );
    strokeShape.moveToPoint( point );
    point = point.plus( Vector2.X_UNIT.times( miterMagnitude ) );
    strokeShape.lineToPoint( point );
    point = point.plus( Vector2.createPolar( miterMagnitude, miterAngleRadians ).negated() );
    strokeShape.lineToPoint( point );
    const fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );
} );

QUnit.test( 'Overlapping rectangles', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;

  const strokeShape = new Shape();
  strokeShape.rect( 40, 40, 100, 100 );
  strokeShape.rect( 50, 50, 100, 100 );
  strokeShape.rect( 80, 80, 100, 100 );
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );

QUnit.test( 'Bezier Offset', assert => {
  const styles = new LineStyles();
  styles.lineWidth = 30;

  const strokeShape = new Shape();
  strokeShape.moveTo( 40, 40 );
  strokeShape.quadraticCurveTo( 100, 200, 160, 40 );
  // strokeShape.close();
  const fillShape = strokeShape.getStrokedShape( styles );

  strokeEqualsFill( assert, strokeShape, fillShape, node => { node.setLineStyles( styles ); }, QUnit.config.current.testName );
} );