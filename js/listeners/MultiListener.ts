// Copyright 2017-2025, University of Colorado Boulder

/**
 * MultiListener is responsible for monitoring the mouse, touch, and other presses on the screen and determines the
 * operations to apply to a target Node from this input. Single touch dragging on the screen will initiate
 * panning. Multi-touch gestures will initiate scaling, translation, and potentially rotation depending on
 * the gesture.
 *
 * MultiListener will keep track of all "background" presses on the screen. When certain conditions are met, the
 * "background" presses become active and attached listeners may be interrupted so that the MultiListener
 * gestures take precedence. MultiListener uses the Intent feature of Pointer, so that the default behavior of this
 * listener can be prevented if necessary. Generally, you would use Pointer.reserveForDrag() to indicate
 * that your Node is intended for other input that should not be interrupted by this listener.
 *
 * For example usage, see scenery/examples/input.html. A typical "simple" MultiListener usage
 * would be something like:
 *
 *    display.addInputListener( new PressListener( targetNode ) );
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

import Property from '../../../axon/js/Property.js';
import Matrix from '../../../dot/js/Matrix.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import { SingularValueDecomposition } from '../../../dot/js/Matrix.js';
import Vector2 from '../../../dot/js/Vector2.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import optionize from '../../../phet-core/js/optionize.js';
import { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { Intent } from '../input/Pointer.js';
import Mouse from '../input/Mouse.js';
import MultiListenerPress from '../listeners/MultiListenerPress.js';
import Node from '../nodes/Node.js';
import Pointer from '../input/Pointer.js';
import scenery from '../scenery.js';
import SceneryEvent from '../input/SceneryEvent.js';
import type TInputListener from '../input/TInputListener.js';

// constants
// pointer must move this much to initiate a move interruption for panning, in the global coordinate frame
const MOVE_INTERRUPT_MAGNITUDE = 25;

export type MultiListenerOptions = {

  // {number} - Restricts input to the specified mouse button (but allows any touch). Only one mouse button is
  // allowed at a time. The button numbers are defined in https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button,
  // where typically:
  //   0: Left mouse button
  //   1: Middle mouse button (or wheel press)
  //   2: Right mouse button
  //   3+: other specific numbered buttons that are more rare
  mouseButton?: number;

  // {string} - Sets the Pointer cursor to this cursor when the listener is "pressed".
  pressCursor?: string;

  // {boolean} - If true, the listener will scale the targetNode from input
  allowScale?: boolean;

  // {boolean} - If true, the listener will rotate the targetNode from input
  allowRotation?: boolean;

  // {boolean} - if true, multitouch will interrupt any active pointer listeners and initiate translation
  // and scale from multitouch gestures
  allowMultitouchInterruption?: boolean;

  // if true, a certain amount of movement in the global coordinate frame will interrupt any pointer listeners and
  // initiate translation from the pointer, unless default behavior has been prevented by setting Intent on the Pointer.
  allowMoveInterruption?: boolean;

  // magnitude limits for scaling in both x and y
  minScale?: number;
  maxScale?: number;
} & Pick<PhetioObjectOptions, 'tandem'>;

class MultiListener implements TInputListener {

  // the Node that will be transformed by this listener
  protected readonly _targetNode: Node;

  // see options
  protected readonly _minScale: number;
  protected readonly _maxScale: number;
  private readonly _mouseButton: number;
  protected readonly _pressCursor: string;
  private readonly _allowScale: boolean;
  private readonly _allowRotation: boolean;
  private readonly _allowMultitouchInterruption: boolean;
  private readonly _allowMoveInterruption: boolean;

  // List of "active" Presses down from Pointer input which are actively changing the transformation of the target Node
  protected readonly _presses: MultiListenerPress[];

  // List of "background" presses which are saved but not yet doing anything for the target Node transformation. If
  // the Pointer already has listeners, Presses are added to the background and wait to be converted to "active"
  // presses until we are allowed to interrupt the other listeners. Related to options "allowMoveInterrupt" and
  // "allowMultitouchInterrupt", where other Pointer listeners are interrupted in these cases.
  private readonly _backgroundPresses: MultiListenerPress[];

  // {Property.<Matrix3>} - The matrix applied to the targetNode in response to various input for the MultiListener
  public readonly matrixProperty: Property<Matrix3>;

  // Whether the listener was interrupted, in which case we may need to prevent certain behavior. If the listener was
  // interrupted, pointer listeners might still be called since input is dispatched to a defensive copy of the
  // Pointer's listeners. But presses will have been cleared in this case so we won't try to do any work on them.
  private _interrupted: boolean;

  // attached to the Pointer when a Press is added
  private _pressListener: TInputListener;

  // attached to the Pointer when presses are added - waits for certain conditions to be met before converting
  // background presses to active presses to enable multitouch listener behavior
  private _backgroundListener: TInputListener;

  /**
   * @constructor
   *
   * @param targetNode - The Node that should be transformed by this MultiListener.
   * @param [providedOptions]
   */
  public constructor( targetNode: Node, providedOptions?: MultiListenerOptions ) {
    const options = optionize<MultiListenerOptions>()( {
      mouseButton: 0,
      pressCursor: 'pointer',
      allowScale: true,
      allowRotation: true,
      allowMultitouchInterruption: false,
      allowMoveInterruption: true,
      minScale: 1,
      maxScale: 4,
      tandem: Tandem.REQUIRED
    }, providedOptions );

    this._targetNode = targetNode;
    this._minScale = options.minScale;
    this._maxScale = options.maxScale;
    this._mouseButton = options.mouseButton;
    this._pressCursor = options.pressCursor;
    this._allowScale = options.allowScale;
    this._allowRotation = options.allowRotation;
    this._allowMultitouchInterruption = options.allowMultitouchInterruption;
    this._allowMoveInterruption = options.allowMoveInterruption;
    this._presses = [];
    this._backgroundPresses = [];

    this.matrixProperty = new Property( targetNode.matrix.copy(), {
      phetioValueType: Matrix3.Matrix3IO,
      tandem: options.tandem.createTandem( 'matrixProperty' ),
      phetioReadOnly: true
    } );

    // assign the matrix to the targetNode whenever it changes
    this.matrixProperty.link( matrix => {
      this._targetNode.matrix = matrix;
    } );

    this._interrupted = false;

    this._pressListener = {
      move: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer move' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        const press = this.findPress( event.pointer )!;
        assert && assert( press, 'Press should be found for move event' );
        this.movePress( press );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      up: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer up' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        const press = this.findPress( event.pointer )!;
        assert && assert( press, 'Press should be found for up event' );
        this.removePress( press );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      cancel: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener pointer cancel' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        const press = this.findPress( event.pointer )!;
        assert && assert( press, 'Press should be found for cancel event' );
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

        if ( !this._interrupted ) {
          const backgroundPress = this.findBackgroundPress( event.pointer )!;
          assert && assert( backgroundPress, 'Background press should be found for up event' );
          this.removeBackgroundPress( backgroundPress );
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      move: event => {

        // Any background press needs to meet certain conditions to be promoted to an actual press that pans/zooms
        const candidateBackgroundPresses = this._backgroundPresses.filter( press => {

          // Dragged pointers and pointers that haven't moved a certain distance are not candidates, and should not be
          // interrupted. We don't want to interrupt taps that might move a little bit
          return !press.pointer.hasIntent( Intent.DRAG ) && press.initialPoint.distance( press.pointer.point ) > MOVE_INTERRUPT_MAGNITUDE;
        } );

        // If we are already zoomed in, we should promote any number of background presses to actual presses.
        // Otherwise, we'll need at least two presses to zoom
        // It is nice to allow down pointers to move around freely without interruption when there isn't any zoom,
        // but we still allow interruption if the number of background presses indicate the user is trying to
        // zoom in
        if ( this.getCurrentScale() !== 1 || candidateBackgroundPresses.length >= 2 ) {
          sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener attached, interrupting for press' );

          // Convert all candidate background presses to actual presses
          candidateBackgroundPresses.forEach( press => {
            this.removeBackgroundPress( press );
            this.interruptOtherListeners( press.pointer );
            this.addPress( press );
          } );
        }
      },

      cancel: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener background cancel' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        if ( !this._interrupted ) {
          const backgroundPress = this.findBackgroundPress( event.pointer )!;
          assert && assert( backgroundPress, 'Background press should be found for cancel event' );
          this.removeBackgroundPress( backgroundPress );
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      interrupt: () => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener background interrupt' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.interrupt();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    };
  }

  /**
   * Finds a Press by searching for the one with the provided Pointer.
   */
  private findPress( pointer: Pointer ): MultiListenerPress | null {
    for ( let i = 0; i < this._presses.length; i++ ) {
      if ( this._presses[ i ].pointer === pointer ) {
        return this._presses[ i ];
      }
    }
    return null;
  }

  /**
   * Find a background Press by searching for one with the provided Pointer. A background Press is one created
   * when we receive an event while a Pointer is already attached.
   */
  private findBackgroundPress( pointer: Pointer ): MultiListenerPress | null {
    for ( let i = 0; i < this._backgroundPresses.length; i++ ) {
      if ( this._backgroundPresses[ i ].pointer === pointer ) {
        return this._backgroundPresses[ i ];
      }
    }
    return null;
  }

  /**
   * Returns true if the press is already contained in one of this._backgroundPresses or this._presses. There are cases
   * where we may try to add the same pointer twice (user opened context menu, using a mouse during fuzz testing), and
   * we want to avoid adding a press again in those cases.
   */
  private hasPress( press: MultiListenerPress ): boolean {
    return _.some( this._presses.concat( this._backgroundPresses ), existingPress => {
      return existingPress.pointer === press.pointer;
    } );
  }

  /**
   * Interrupt all listeners on the pointer, except for background listeners that
   * were added by this MultiListener. Useful when it is time for this listener to
   * "take over" and interrupt any other listeners on the pointer.
   */
  private interruptOtherListeners( pointer: Pointer ): void {
    const listeners = pointer.getListeners();
    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];
      if ( listener !== this._backgroundListener ) {
        listener.interrupt && listener.interrupt();
      }
    }
  }

  /**
   * Part of the scenery event API. (scenery-internal)
   */
  public down( event: SceneryEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener down' );

    if ( event.pointer instanceof Mouse && event.domEvent instanceof MouseEvent && event.domEvent.button !== this._mouseButton ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener abort: wrong mouse button' );
      return;
    }

    // clears the flag for MultiListener behavior
    this._interrupted = false;

    let pressTrail;
    if ( !_.includes( event.trail.nodes, this._targetNode ) ) {

      // if the target Node is not in the event trail, we assume that the event went to the
      // Display or the root Node of the scene graph - this will throw an assertion if
      // there are more than one trails found
      pressTrail = this._targetNode.getUniqueTrailTo( event.target );
    }
    else {
      pressTrail = event.trail.subtrailTo( this._targetNode, false );
    }
    assert && assert( _.includes( pressTrail.nodes, this._targetNode ), 'targetNode must be in the Trail for Press' );

    sceneryLog && sceneryLog.InputListener && sceneryLog.push();
    const press = new MultiListenerPress( event.pointer, pressTrail );

    if ( !this._allowMoveInterruption && !this._allowMultitouchInterruption ) {

      // most restrictive case, only allow presses if the pointer is not attached - Presses
      // are never added as background presses in this case because interruption is never allowed
      if ( !event.pointer.isAttached() ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener unattached, using press' );
        this.addPress( press );
      }
    }
    else {

      // we allow some form of interruption, add as background presses, and we will decide if they
      // should be converted to presses and interrupt other listeners on move event
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener attached, adding background press' );
      this.addBackgroundPress( press );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Add a Press to this listener when a new Pointer is down.
   */
  protected addPress( press: MultiListenerPress ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener addPress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    if ( !this.hasPress( press ) ) {
      this._presses.push( press );

      press.pointer.cursor = this._pressCursor;
      press.pointer.addInputListener( this._pressListener, true );

      this.recomputeLocals();
      this.reposition();
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Reposition in response to movement of any Presses.
   */
  protected movePress( press: MultiListenerPress ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener movePress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.reposition();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Remove a Press from this listener.
   */
  protected removePress( press: MultiListenerPress ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener removePress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    press.pointer.removeInputListener( this._pressListener );
    press.pointer.cursor = null;

    arrayRemove( this._presses, press );

    this.recomputeLocals();
    this.reposition();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Add a background Press, a Press that we receive while a Pointer is already attached. Depending on background
   * Presses, we may interrupt the attached pointer to begin zoom operations.
   */
  private addBackgroundPress( press: MultiListenerPress ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener addBackgroundPress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // It's possible that the press pointer already has the listener - for instance in Chrome we fail to get
    // "up" events once the context menu is open (like after a right click), so only add to the Pointer
    // if it isn't already added
    if ( !this.hasPress( press ) ) {
      this._backgroundPresses.push( press );
      press.pointer.addInputListener( this._backgroundListener, false );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Remove a background Press from this listener.
   */
  private removeBackgroundPress( press: MultiListenerPress ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener removeBackgroundPress' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    press.pointer.removeInputListener( this._backgroundListener );

    arrayRemove( this._backgroundPresses, press );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Reposition the target Node (including all apsects of transformation) of this listener's target Node.
   */
  protected reposition(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener reposition' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.matrixProperty.set( this.computeMatrix() );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Recompute the local points of the Presses for this listener, relative to the target Node.
   */
  protected recomputeLocals(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener recomputeLocals' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    for ( let i = 0; i < this._presses.length; i++ ) {
      this._presses[ i ].recomputeLocalPoint();
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Interrupt this listener.
   */
  public interrupt(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'MultiListener interrupt' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    while ( this._presses.length ) {
      this.removePress( this._presses[ this._presses.length - 1 ] );
    }

    while ( this._backgroundPresses.length ) {
      this.removeBackgroundPress( this._backgroundPresses[ this._backgroundPresses.length - 1 ] );
    }

    this._interrupted = true;

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Compute the transformation matrix for the target Node based on Presses.
   */
  private computeMatrix(): Matrix3 {
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

  /**
   * Compute a transformation matrix from a single press. Single press indicates translation (panning) for the
   * target Node.
   */
  private computeSinglePressMatrix(): Matrix3 {
    const singleTargetPoint = this._presses[ 0 ].targetPoint;
    const localPoint = this._presses[ 0 ].localPoint!;
    assert && assert( localPoint, 'localPoint is not defined on the Press?' );

    const singleMappedPoint = this._targetNode.localToParentPoint( localPoint );
    const delta = singleTargetPoint.minus( singleMappedPoint );
    return Matrix3.translationFromVector( delta ).timesMatrix( this._targetNode.getMatrix() );
  }

  /**
   * Compute a translation matrix from multiple presses. Usually multiple presses will have some scale or rotation
   * as well, but this is to be used if rotation and scale are not enabled for this listener.
   */
  public computeTranslationMatrix(): Matrix3 {
    // translation only. linear least-squares simplifies to sum of differences
    const sum = new Vector2( 0, 0 );
    for ( let i = 0; i < this._presses.length; i++ ) {
      sum.add( this._presses[ i ].targetPoint );

      const localPoint = this._presses[ i ].localPoint!;
      assert && assert( localPoint, 'localPoint is not defined on the Press?' );
      sum.subtract( localPoint );
    }
    return Matrix3.translationFromVector( sum.dividedScalar( this._presses.length ) );
  }

  /**
   * A transformation matrix from multiple Presses that will translate and scale the target Node.
   */
  private computeTranslationScaleMatrix(): Matrix3 {
    const localPoints = this._presses.map( press => {
      assert && assert( press.localPoint, 'localPoint is not defined on the Press?' );
      return press.localPoint!;
    } );
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

    // while fuzz testing, it is possible that the Press points are
    // exactly the same resulting in undefined scale - if that is the case
    // we will not adjust
    let scale = this.getCurrentScale();
    if ( targetSquaredDistance !== 0 ) {
      scale = this.limitScale( Math.sqrt( targetSquaredDistance / localSquaredDistance ) );
    }

    const translateToTarget = Matrix3.translation( targetCentroid.x, targetCentroid.y );
    const translateFromLocal = Matrix3.translation( -localCentroid.x, -localCentroid.y );

    return translateToTarget.timesMatrix( Matrix3.scaling( scale ) ).timesMatrix( translateFromLocal );
  }

  /**
   * Limit the provided scale by constraints of this MultiListener.
   */
  protected limitScale( scale: number ): number {
    let correctedScale = Math.max( scale, this._minScale );
    correctedScale = Math.min( correctedScale, this._maxScale );
    return correctedScale;
  }

  /**
   * Compute a transformation matrix that will translate and scale the target Node from multiple presses. Should
   * be used when scaling is not enabled for this listener.
   */
  private computeTranslationRotationMatrix(): Matrix3 {
    let i;
    const localMatrix = new Matrix( 2, this._presses.length );
    const targetMatrix = new Matrix( 2, this._presses.length );
    const localCentroid = new Vector2( 0, 0 );
    const targetCentroid = new Vector2( 0, 0 );
    for ( i = 0; i < this._presses.length; i++ ) {
      const localPoint = this._presses[ i ].localPoint!;
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

  /**
   * Compute a transformation matrix that will translate, scale, and rotate the target Node from multiple Presses.
   */
  private computeTranslationRotationScaleMatrix(): Matrix3 {
    let i;
    const localMatrix = new Matrix( this._presses.length * 2, 4 );
    for ( i = 0; i < this._presses.length; i++ ) {
      // [ x  y 1 0 ]
      // [ y -x 0 1 ]
      const localPoint = this._presses[ i ].localPoint!;
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
   * Get the current scale on the target Node, assumes that there is isometric scaling in both x and y.
   */
  public getCurrentScale(): number {
    return this._targetNode.getScaleVector().x;
  }

  /**
   * Reset transform on the target Node.
   */
  public resetTransform(): void {
    this._targetNode.resetTransform();
    this.matrixProperty.set( this._targetNode.matrix.copy() );
  }
}

scenery.register( 'MultiListener', MultiListener );

export default MultiListener;