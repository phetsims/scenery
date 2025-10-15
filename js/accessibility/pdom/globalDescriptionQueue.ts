// Copyright 2025, University of Colorado Boulder

/**
 * Global queue for Interactive Description responses that are not tied to a visible `Node`.
 *
 * Initialize this queue with the application's aria-live container and use it to announce results of interactions
 * where the relevant `Node` has been removed or hidden (for example, a button that disappears immediately after it is
 * pressed). The same filtering applied by the `ParallelDOM` response helpers is preserved so announcements stay
 * consistent with the rest of the accessibility infrastructure.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import isSettingPhetioStateProperty from '../../../../tandem/js/isSettingPhetioStateProperty.js';
import AriaLiveAnnouncer from '../../../../utterance-queue/js/AriaLiveAnnouncer.js';
import Utterance, { TAlertable } from '../../../../utterance-queue/js/Utterance.js';
import UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import scenery from '../../scenery.js';
import Features from '../../util/Features.js';

class GlobalDescriptionQueue extends UtteranceQueue<AriaLiveAnnouncer> {
  private _globalInitialized = false;

  /**
   * Create the queue with its own announcer so announcements are isolated from other queues.
   */
  public constructor() {
    const announcer = new AriaLiveAnnouncer();
    super( announcer );
  }

  /**
   * Attach the aria-live container managed by this queue to the DOM so announcements can be heard.
   *
   * @param parentElement - Element that will host the aria-live container
   */
  public initialize( parentElement: HTMLElement ): void {
    assert && assert( !this._globalInitialized, 'Already initialized, cannot initialize again.' );
    parentElement.appendChild( this.announcer.ariaLiveContainer );

    // set `user-select: none` on the aria-live container to prevent iOS text selection issue, see
    // https://github.com/phetsims/scenery/issues/1006
    // @ts-expect-error
    this.announcer.ariaLiveContainer.style[ Features.userSelect ] = 'none';
    this._globalInitialized = true;
  }

  /**
   * Enqueue an accessible response that will be announced through this queue's aria-live container.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleResponse( alertable: TAlertable, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    assert && assert( this._globalInitialized, 'Must be initialized to use.' );

    // Nothing to do if there is no content.
    if ( alertable === null || Utterance.alertableToText( alertable ) === '' ) {
      return;
    }

    // No description should be alerted if setting PhET-iO state, see https://github.com/phetsims/scenery/issues/1397
    if ( isSettingPhetioStateProperty.value ) {
      return;
    }

    if ( alertBehavior === 'interrupt' ) {
      this.clear();
    }

    this.addToBack( alertable );
  }

  /**
   * Announce a contextual description, typically describing the result of an interaction.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleContextResponse( alertable: TAlertable, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    this.addAccessibleResponse( alertable, alertBehavior );
  }

  /**
   * Announce an object description, typically describing what an object is or its state.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleObjectResponse( alertable: TAlertable, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    this.addAccessibleResponse( alertable, alertBehavior );
  }

  /**
   * Announce a help response, typically providing instructions or guidance.
   *
   * @param alertable - The content to be announced by screen readers
   * @param alertBehavior - Controls whether the response interrupts existing ones ('interrupt') or waits in the queue ('queue')
   */
  public addAccessibleHelpResponse( alertable: TAlertable, alertBehavior: 'queue' | 'interrupt' = 'interrupt' ): void {
    this.addAccessibleResponse( alertable, alertBehavior );
  }
}

const globalDescriptionQueue = new GlobalDescriptionQueue();

scenery.register( 'globalDescriptionQueue', globalDescriptionQueue );

export default globalDescriptionQueue;
