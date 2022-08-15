// Copyright 2015-2022, University of Colorado Boulder

/**
 * Used for identifying when any ancestor transform of a node in a trail causes that node's global transform to change.
 * It also provides fast computation of that global matrix, NOT recomputing every matrix, even on most transform
 * changes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import optionize from '../../../phet-core/js/optionize.js';
import { scenery, Trail } from '../imports.js';

type SelfOptions = {
  // Whether the bounds listeners should be added with on() or onStatic().
  isStatic?: boolean;
};

export type TransformTrackerOptions = SelfOptions;

class TransformTracker {

  private readonly trail: Trail;

  // this._matrices[ i ] will be equal to: trail.nodes[ 1 ].matrix * ... * trail.nodes[ i + 1 ].matrix
  // Will be initialized on first need.
  private _matrices: Matrix3[] | null = null;

  // this._matrices[ i ] where i >= this._dirtyIndex will need to be recomputed
  private _dirtyIndex = 0;

  // Listeners added by client, will be called on transform changes.
  private _listeners: ( () => void )[] = [];

  // Listeners to each Node in the trail (so we are notified of changes). Will be removed on disposal.
  private readonly _nodeTransformListeners: ( () => void )[] = [];

  private readonly _isStatic: boolean;

  /**
   * Creates a transform-tracking object, where it can send out updates on transform changes, and also efficiently
   * compute the transform.
   */
  public constructor( trail: Trail, providedOptions?: TransformTrackerOptions ) {

    const options = optionize<TransformTrackerOptions, SelfOptions>()( {
      isStatic: false
    }, providedOptions );

    this._isStatic = options.isStatic;
    this.trail = trail;

    // Hook up listeners to each Node in the trail
    this._nodeTransformListeners = [];
    for ( let j = 1; j < this.trail.length; j++ ) {
      // Wrapping with closure to prevent changes
      const nodeTransformListener = ( index => () => {
        this.onTransformChange( index );
      } )( j - 1 );

      this._nodeTransformListeners.push( nodeTransformListener );

      trail.nodes[ j ].transformEmitter.addListener( nodeTransformListener );
    }
  }

  /**
   * Gets rid of all external references and listeners. This object is inoperable afterwards.
   */
  public dispose(): void {
    for ( let j = 1; j < this.trail.length; j++ ) {
      const nodeTransformListener = this._nodeTransformListeners[ j - 1 ];

      if ( this.trail.nodes[ j ].transformEmitter.hasListener( nodeTransformListener ) ) {
        this.trail.nodes[ j ].transformEmitter.removeListener( nodeTransformListener );
      }
    }
  }

  /**
   * Adds a listener function that will be synchronously called whenever the transform for this Trail changes.
   */
  public addListener( listener: () => void ): void {
    this._listeners.push( listener );
  }

  /**
   * Removes a listener that was previously added with addListener().
   */
  public removeListener( listener: () => void ): void {
    const index = _.indexOf( this._listeners, listener );
    assert && assert( index >= 0, 'TransformTracker listener not found' );

    this._listeners.splice( index, 1 );
  }

  /**
   * Notifies listeners of a transform change.
   */
  private notifyListeners(): void {
    let listeners = this._listeners;

    if ( !this._isStatic ) {
      listeners = listeners.slice();
    }

    const length = listeners.length;
    for ( let i = 0; i < length; i++ ) {
      listeners[ i ]();
    }
  }

  /**
   * Called when one of the nodes' transforms is changed.
   *
   * @param matrixIndex - The index into our matrices array, e.g. this._matrices[ matrixIndex ].
   */
  private onTransformChange( matrixIndex: number ): void {
    this._dirtyIndex = Math.min( this._dirtyIndex, matrixIndex );
    this.notifyListeners();
  }

  /**
   * Returns the local-to-global transformation matrix for the Trail, which transforms its leaf node's local
   * coordinate frame into the global coordinate frame.
   *
   * NOTE: The matrix returned should not be mutated. Please make a copy if needed.
   */
  public getMatrix(): Matrix3 {
    if ( this._matrices === null ) {
      this._matrices = [];

      // Start at 1, so that we don't include the root node's transform
      for ( let i = 1; i < this.trail.length; i++ ) {
        this._matrices.push( new Matrix3() );
      }
    }

    // If the trail isn't long enough to have a transform, return the identity matrix
    if ( this._matrices.length <= 0 ) {
      return Matrix3.IDENTITY;
    }

    // Starting at the dirty index, recompute matrices.
    const numMatrices = this._matrices.length;
    for ( let index = this._dirtyIndex; index < numMatrices; index++ ) {
      const nodeMatrix = this.trail.nodes[ index + 1 ].matrix;

      if ( index === 0 ) {
        this._matrices[ index ].set( nodeMatrix );
      }
      else {
        this._matrices[ index ].set( this._matrices[ index - 1 ] );
        this._matrices[ index ].multiplyMatrix( nodeMatrix );
      }
    }

    // Reset the dirty index to mark all matrices as 'clean'.
    this._dirtyIndex = numMatrices;

    // Return the last matrix, which contains our composite transformation.
    return this._matrices[ numMatrices - 1 ];
  }

  public get matrix(): Matrix3 { return this.getMatrix(); }
}

scenery.register( 'TransformTracker', TransformTracker );
export default TransformTracker;