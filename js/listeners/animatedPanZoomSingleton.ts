// Copyright 2021-2025, University of Colorado Boulder

/**
 * A singleton with an instance of an AnimatedPanZoomListener that can be easily accessed for an application.
 * The AnimatedPanZoomListener adds global key press listeners and it is reasonable that you would only want
 * one for a document/application.
 *
 * @author Jesse Greenberg
 */

import AnimatedPanZoomListener, { AnimatedPanZoomListenerOptions } from '../listeners/AnimatedPanZoomListener.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';

class AnimatedPanZoomSingleton {

  // A reference to the instance of the listener, null until initialized.
  private _listener: AnimatedPanZoomListener | null = null;

  public initialize( targetNode: Node, options?: AnimatedPanZoomListenerOptions ): void {
    this._listener = new AnimatedPanZoomListener( targetNode, options );
  }

  public dispose(): void {
    assert && assert( this._listener, 'No listener, call initialize first.' );
    this._listener!.dispose();
    this._listener = null;
  }

  /**
   * Returns the AnimatedPanZoomListener.
   */
  public get listener(): AnimatedPanZoomListener {
    assert && assert( this._listener, 'No listener, call initialize first.' );
    return this._listener!;
  }

  /**
   * Returns true if the animatedPanZoomSingleton has been initialized.
   */
  public get initialized(): boolean {
    return !!this._listener;
  }
}

const animatedPanZoomSingleton = new AnimatedPanZoomSingleton();
scenery.register( 'animatedPanZoomSingleton', animatedPanZoomSingleton );
export default animatedPanZoomSingleton;