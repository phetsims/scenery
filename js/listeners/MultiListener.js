// Copyright 2017-2020, University of Colorado Boulder

/**
 * NOTE: Not a fully finished product, please BEWARE before using this in code.
 *
 * TODO: doc
 *
 * TODO: unit tests
 *
 * TODO: add example usage
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix from '../../../dot/js/Matrix.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import SingularValueDecomposition from '../../../dot/js/SingularValueDecomposition.js';
import Vector2 from '../../../dot/js/Vector2.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import merge from '../../../phet-core/js/merge.js';
import Mouse from '../input/Mouse.js';
import Pointer from '../input/Pointer.js';
import scenery from '../scenery.js';

// constants
// pointer must move this much to initiate a move interruption for panning, in the global coordinate frame
const MOVE_INTERRUPT_MAGNITUDE = 25;

class MultiListener {

  /**
   * @constructor
   *
   * @param {Node} targetNode - The Node that should be transformed by this MultiListener.
   * @param {Object} [options] - See the constructor body (below) for documented options.
   */
  constructor( targetNode, options ) {

    options = merge( {
      mouseButton: 0, // TODO: see PressListener
      pressCursor: 'pointer', // TODO: see PressListener
      targetNode: null, // TODO: required? pass in at front
      allowScale: true,
      allowRotation: true,

      // {boolean} - if true, multitouch will interrupt any active pointer listeners and and initiate translation
      // and scale from multitouch gestures
      allowMultitouchInterruption: false,

      // {private} - if true, a certain amount of movement in the global coordinate frame with interrupt any pointer
      // listeners and initiate translation from the pointer
      allowMoveInterruption: true,

      // {number} - limits for scaling
      minScale: 1,
      maxScale: 4
    }, options );

    // TODO: type checks for options

    this._targetNode = targetNode;

    // @protected (read-only)
    this._minScale = options.minScale;
    this._maxScale = options.maxScale;

    // @private - see options
    this._mouseButton = options.mouseButton;
    this._pressCursor = options.pressCursor;
    this._allowScale = options.allowScale;
    this._allowRotation = options.allowRotation;
    this._allowMultitouchInterruption = options.allowMultitouchInterruption;
    this._allowMoveInterruption = options.allowMoveInterruption;

    // @private {Array.<Press>}
    this._presses = [];

    // @private {Array.<Press>}
    this._backgroundPresses = [];

    // @private
    this._pressListener = {
      move: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer move' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.movePress( this.findPress( event.pointer ) );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      up: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer up' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        // TODO: consider logging press on the pointer itself?
        this.removePress( this.findPress( event.pointer ) );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      cancel: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer cancel' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        const press = this.findPress( event.pointer );
        press.interrupted = true;

        this.removePress( press );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      interrupt: () => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer interrupt' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        // For the future, we could figure out how to track the pointer that calls this
        this.interrupt();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    };

    this._backgroundListener = {
      up: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener background up' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.removeBackgroundPress( this.findBackgroundPress( event.pointer ) );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      move: event => {
        if ( this._allowMoveInterruption && this.canMove( event.pointer ) ) {

          const backgroundPress = this.findBackgroundPress( event.pointer );

          // TODO: scratch Vector
          const difference = backgroundPress.initialPoint.minus( event.pointer.point );

          // it is nice to allow down pointers to move around freely without interruption when there isn't any zoom,
          // but we still allow interruption if the number of background presses indicate the user is trying to
          // zoom in
          const zoomedOutPan = this._backgroundPresses.length <= 1 && this.getCurrentScale() === 1;
          if ( difference.magnitude > MOVE_INTERRUPT_MAGNITUDE && !zoomedOutPan ) {

            // only interrupt if pointer has moved far enough so we don't interrupt taps that might move little or
            // when there is no scale to translate
            sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener attached, interrupting for press' );
            event.pointer.interruptAttached();
            this.convertBackgroundPresses();
            this.movePress( this.findPress( event.pointer ) );
          }
        }
      },

      cancel: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener background cancel' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.removeBackgroundPress( this.findBackgroundPress( event.pointer ) );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      interrupt: () => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener background interrupt' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.removeBackgroundPress( this.findBackgroundPress( event.pointer ) );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    };
  }

  findPress( pointer ) {
    for ( let i = 0; i < this._presses.length; i++ ) {
      if ( this._presses[ i ].pointer === pointer ) {
        return this._presses[ i ];
      }
    }
    assert && assert( false, 'Did not find press' );
    return null;
  }

  findBackgroundPress( pointer ) {
    // TODO: reduce duplication with findPress?
    for ( let i = 0; i < this._backgroundPresses.length; i++ ) {
      if ( this._backgroundPresses[ i ].pointer === pointer ) {
        return this._backgroundPresses[ i ];
      }
    }
    assert && assert( false, 'Did not find press' );
    return null;
  }

  /**
   * Movement is disallowed for pointers with Intent that indicate that dragging is expected.
   * @private
   *
   * @param {Pointer} pointer
   * @returns {boolean}
   */
  canMove( pointer ) {
    const intent = pointer.getIntent();
    return ( intent !== Pointer.Intent.DRAG && intent !== Pointer.Intent.MULTI_DRAG );
  }

  // TODO: see PressListener
  down( event ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener down' );

    if ( event.pointer instanceof Mouse && event.domEvent.button !== this._mouseButton ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener abort: wrong mouse button' );

      return;
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    assert && assert( _.includes( event.trail.nodes, this._targetNode ),
      'MultiListener down trail does not include targetNode?' );

    const press = new Press( event.pointer, event.trail.subtrailTo( this._targetNode, false ) );

    if ( !event.pointer.isAttached() ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener unattached, using press' );
      this.addPress( press );
      this.convertBackgroundPresses();
    }
    else if ( this._allowMultitouchInterruption ) {
      if ( this._presses.length || this._backgroundPresses.length ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener attached, interrupting for press' );
        press.pointer.interruptAttached();
        this.addPress( press );
        this.convertBackgroundPresses();
      }
      else {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener attached, adding background press' );
        this.addBackgroundPress( press );
      }
    }
    else if ( this._allowMoveInterruption ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener attached, adding background press for move' );
      this.addBackgroundPress( press );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  addPress( press ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener addPress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._presses.push( press );

    press.pointer.cursor = this._pressCursor;
    press.pointer.addInputListener( this._pressListener, true );

    this.recomputeLocals();
    this.reposition();

    // TODO: handle interrupted?

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  movePress( press ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener movePress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.reposition();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  removePress( press ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener removePress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    press.pointer.removeInputListener( this._pressListener );
    press.pointer.cursor = null;

    arrayRemove( this._presses, press );

    this.recomputeLocals();
    this.reposition();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  addBackgroundPress( press ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener addBackgroundPress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // TODO: handle turning background presses into main presses here
    this._backgroundPresses.push( press );
    press.pointer.addInputListener( this._backgroundListener, false );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  removeBackgroundPress( press ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener removeBackgroundPress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    press.pointer.removeInputListener( this._backgroundListener );

    arrayRemove( this._backgroundPresses, press );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  convertBackgroundPresses() {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener convertBackgroundPresses' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    const presses = this._backgroundPresses.slice();
    for ( let i = 0; i < presses.length; i++ ) {
      const press = presses[ i ];
      this.removeBackgroundPress( press );
      press.pointer.interruptAttached();
      this.addPress( press );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  reposition() {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener reposition' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this._targetNode.matrix = this.computeMatrix();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  recomputeLocals() {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener recomputeLocals' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    for ( let i = 0; i < this._presses.length; i++ ) {
      this._presses[ i ].recomputeLocalPoint();
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  interrupt() {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener interrupt' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    while ( this._presses.length ) {
      this.removePress( this._presses[ this._presses.length - 1 ] );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  // @private?
  computeMatrix() {
    if ( this._presses.length === 0 ) {
      return this._targetNode.getMatrix();
    }
    else if ( this._presses.length === 1 ) {
      return this.computeSinglePressMatrix();
    }
    else if ( this._allowScale && this._allowRotation ) {
      return this.computeTranslationRotationScaleMatrix();
    }
    else if ( this._allowScale ) {
      return this.computeTranslationScaleMatrix();
    }
    else if ( this._allowRotation ) {
      return this.computeTranslationRotationMatrix();
    }
    else {
      return this.computeTranslationMatrix();
    }
  }

  // @private
  computeSinglePressMatrix() {
    // TODO: scratch things
    const singleTargetPoint = this._presses[ 0 ].targetPoint;
    const singleMappedPoint = this._targetNode.localToParentPoint( this._presses[ 0 ].localPoint );
    const delta = singleTargetPoint.minus( singleMappedPoint );
    return Matrix3.translationFromVector( delta ).timesMatrix( this._targetNode.getMatrix() );
  }

  // @private
  computeTranslationMatrix() {
    // translation only. linear least-squares simplifies to sum of differences
    const sum = new Vector2( 0, 0 );
    for ( let i = 0; i < this._presses.length; i++ ) {
      sum.add( this._presses[ i ].targetPoint );
      sum.subtract( this._presses[ i ].localPoint );
    }
    return Matrix3.translationFromVector( sum.dividedScalar( this._presses.length ) );
  }

  // @private
  computeTranslationScaleMatrix() {
    // TODO: minimize closures
    const localPoints = this._presses.map( press => press.localPoint );
    const targetPoints = this._presses.map( press => press.targetPoint );

    const localCentroid = new Vector2( 0, 0 );
    const targetCentroid = new Vector2( 0, 0 );

    localPoints.forEach( localPoint => { localCentroid.add( localPoint ); } );
    targetPoints.forEach( targetPoint => { targetCentroid.add( targetPoint ); } );

    localCentroid.divideScalar( this._presses.length );
    targetCentroid.divideScalar( this._presses.length );

    let localSquaredDistance = 0;
    let targetSquaredDistance = 0;

    localPoints.forEach( localPoint => { localSquaredDistance += localPoint.distanceSquared( localCentroid ); } );
    targetPoints.forEach( targetPoint => { targetSquaredDistance += targetPoint.distanceSquared( targetCentroid ); } );

    let scale = Math.sqrt( targetSquaredDistance / localSquaredDistance );
    scale = this.limitScale( scale );

    const translateToTarget = Matrix3.translation( targetCentroid.x, targetCentroid.y );
    const translateFromLocal = Matrix3.translation( -localCentroid.x, -localCentroid.y );

    return translateToTarget.timesMatrix( Matrix3.scaling( scale ) ).timesMatrix( translateFromLocal );
  }

  /**
   * Limit the provided scale by constraints of this MultiListener.
   *
   * @param {number} scale
   * @returns {number}
   */
  limitScale( scale ) {
    let correctedScale = Math.max( scale, this._minScale );
    correctedScale = Math.min( correctedScale, this._maxScale );
    return correctedScale;
  }

  // @private
  computeTranslationRotationMatrix() {
    let i;
    const localMatrix = new Matrix( 2, this._presses.length );
    const targetMatrix = new Matrix( 2, this._presses.length );
    const localCentroid = new Vector2( 0, 0 );
    const targetCentroid = new Vector2( 0, 0 );
    for ( i = 0; i < this._presses.length; i++ ) {
      const localPoint = this._presses[ i ].localPoint;
      const targetPoint = this._presses[ i ].targetPoint;
      localCentroid.add( localPoint );
      targetCentroid.add( targetPoint );
      localMatrix.set( 0, i, localPoint.x );
      localMatrix.set( 1, i, localPoint.y );
      targetMatrix.set( 0, i, targetPoint.x );
      targetMatrix.set( 1, i, targetPoint.y );
    }
    localCentroid.divideScalar( this._presses.length );
    targetCentroid.divideScalar( this._presses.length );

    // determine offsets from the centroids
    for ( i = 0; i < this._presses.length; i++ ) {
      localMatrix.set( 0, i, localMatrix.get( 0, i ) - localCentroid.x );
      localMatrix.set( 1, i, localMatrix.get( 1, i ) - localCentroid.y );
      targetMatrix.set( 0, i, targetMatrix.get( 0, i ) - targetCentroid.x );
      targetMatrix.set( 1, i, targetMatrix.get( 1, i ) - targetCentroid.y );
    }
    const covarianceMatrix = localMatrix.times( targetMatrix.transpose() );
    const svd = new SingularValueDecomposition( covarianceMatrix );
    let rotation = svd.getV().times( svd.getU().transpose() );
    if ( rotation.det() < 0 ) {
      rotation = svd.getV().times( Matrix.diagonalMatrix( [ 1, -1 ] ) ).times( svd.getU().transpose() );
    }
    const rotation3 = new Matrix3().rowMajor( rotation.get( 0, 0 ), rotation.get( 0, 1 ), 0,
      rotation.get( 1, 0 ), rotation.get( 1, 1 ), 0,
      0, 0, 1 );
    const translation = targetCentroid.minus( rotation3.timesVector2( localCentroid ) );
    rotation3.set02( translation.x );
    rotation3.set12( translation.y );
    return rotation3;
  }

  // @private
  computeTranslationRotationScaleMatrix() {
    let i;
    const localMatrix = new Matrix( this._presses.length * 2, 4 );
    for ( i = 0; i < this._presses.length; i++ ) {
      // [ x  y 1 0 ]
      // [ y -x 0 1 ]
      const localPoint = this._presses[ i ].localPoint;
      localMatrix.set( 2 * i + 0, 0, localPoint.x );
      localMatrix.set( 2 * i + 0, 1, localPoint.y );
      localMatrix.set( 2 * i + 0, 2, 1 );
      localMatrix.set( 2 * i + 1, 0, localPoint.y );
      localMatrix.set( 2 * i + 1, 1, -localPoint.x );
      localMatrix.set( 2 * i + 1, 3, 1 );
    }
    const targetMatrix = new Matrix( this._presses.length * 2, 1 );
    for ( i = 0; i < this._presses.length; i++ ) {
      const targetPoint = this._presses[ i ].targetPoint;
      targetMatrix.set( 2 * i + 0, 0, targetPoint.x );
      targetMatrix.set( 2 * i + 1, 0, targetPoint.y );
    }
    const coefficientMatrix = SingularValueDecomposition.pseudoinverse( localMatrix ).times( targetMatrix );
    const m11 = coefficientMatrix.get( 0, 0 );
    const m12 = coefficientMatrix.get( 1, 0 );
    const m13 = coefficientMatrix.get( 2, 0 );
    const m23 = coefficientMatrix.get( 3, 0 );
    return new Matrix3().rowMajor( m11, m12, m13,
      -m12, m11, m23,
      0, 0, 1 );
  }

  /**
   * Get the current scale on the target node, assumes that there is isometric scaling in both x and y.
   *
   * @public
   * @returns {number}
   */
  getCurrentScale() {
    return this._targetNode.getScaleVector().x;
  }

  /**
   * Reset transform on the target node.
   *
   * @public
   */
  resetTransform() {
    this._targetNode.resetTransform();
  }
}

scenery.register( 'MultiListener', MultiListener );

class Press {
  constructor( pointer, trail ) {
    this.pointer = pointer;
    this.trail = trail;
    this.interrupted = false;

    // @public (read-only) {Vector2} - down point for the new press, in the global coordinate frame
    this.initialPoint = pointer.point;

    this.localPoint = null;
    this.recomputeLocalPoint();
  }

  recomputeLocalPoint() {
    this.localPoint = this.trail.globalToLocalPoint( this.pointer.point );
  }

  get targetPoint() {
    return this.trail.globalToParentPoint( this.pointer.point );
  }
}

export default MultiListener;