// Copyright 2021, University of Colorado Boulder

/**
 * Broadcasts when any Node or its ancestors in a Trail change visibility, effectively
 * observing changes to Trail.isVisible().
 *
 * @author Jesse Greenberg
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import merge from '../../../phet-core/js/merge.js';
import { scenery } from '../imports.js';

class TrailVisibilityTracker {

  /**
   * @param {Trail} trail - the Trail to track visibility
   * @param {Object } [options]
   */
  constructor( trail, options ) {

    options = merge( {

      // {boolan} - whether listeners are called against a defensive copy
      isStatic: false
    }, options );

    // @private {boolean}
    this.isStatic = options.isStatic;

    // @private {Trail}
    this.trail = trail;

    // @private {function[]}
    this._listeners = [];

    // @public {TinyProperty.<boolean>} - True if all Nodes in the Trail are visible.
    this.trailVisibleProperty = new TinyProperty( this.trail.isVisible() );

    // Hook up listeners to each Node in the trail, so we are notified of changes. Will be removed on disposal.
    this._nodeVisibilityListeners = [];
    for ( let j = 0; j < this.trail.length; j++ ) {
      const nodeVisibilityListener = () => {
        this.trailVisibleProperty.set( trail.isVisible() );
      };
      this._nodeVisibilityListeners.push( nodeVisibilityListener );
      trail.nodes[ j ].visibleProperty.link( nodeVisibilityListener );
    }

    this.boundTrailVisibilityChangedListener = this.onVisibilityChange.bind( this );
    this.trailVisibleProperty.link( this.boundTrailVisibilityChangedListener );
  }

  /**
   * Adds a listener function that will be synchronously called whenever the visibility this Trail changes.
   * @public
   *
   * @param {Function} listener - Listener will be called with no arguments.
   */
  addListener( listener ) {
    assert && assert( typeof listener === 'function' );
    this._listeners.push( listener );
  }

  /**
   * Removes a listener that was previously added with addListener().
   * @public
   *
   * @param {Function} listener
   */
  removeListener( listener ) {
    assert && assert( typeof listener === 'function' );

    const index = _.indexOf( this._listeners, listener );
    assert && assert( index >= 0, 'TrailVisibilityTracker listener not found' );

    this._listeners.splice( index, 1 );
  }

  /**
   * @public
   */
  dispose() {
    for ( let j = 0; j < this.trail.length; j++ ) {
      const visibilityListener = this._nodeVisibilityListeners[ j ];

      if ( this.trail.nodes[ j ].visibleProperty.hasListener( visibilityListener ) ) {
        this.trail.nodes[ j ].visibleProperty.removeListener( visibilityListener );
      }
    }

    this.trailVisibleProperty.unlink( this.boundTrailVisibilityChangedListener );
  }

  /**
   * When visibility of the Trail is changed, notify listeners.
   * @private
   */
  onVisibilityChange() {
    this.notifyListeners();
  }

  /**
   * Notify listeners to a change in visibility.
   * @private
   */
  notifyListeners() {
    let listeners = this._listeners;

    if ( !this._isStatic ) {
      listeners = listeners.slice();
    }

    const length = listeners.length;
    for ( let i = 0; i < length; i++ ) {
      listeners[ i ]();
    }
  }
}

scenery.register( 'TrailVisibilityTracker', TrailVisibilityTracker );
export default TrailVisibilityTracker;